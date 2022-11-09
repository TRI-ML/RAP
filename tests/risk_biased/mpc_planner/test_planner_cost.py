from math import sqrt

import pytest
import torch
from mmcv import Config

from risk_biased.mpc_planner.planner_cost import TrackingCost, TrackingCostParams


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    cfg = Config()

    cfg.tracking_cost_scale_longitudinal = 0.1
    cfg.tracking_cost_scale_lateral = 10.0
    cfg.tracking_cost_reduce = "mean"

    return cfg


class TestTrackingCost:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        self.cost_params = TrackingCostParams.from_config(params)
        self.cost = TrackingCost(self.cost_params)

    @pytest.mark.parametrize("target_velocity", [[0.0, 0.0], [1.0, 1.0]])
    def test_get_quadratic_cost_matrix(self, target_velocity):
        target_velocity_trajectory = torch.Tensor(target_velocity).expand(10, 5, -1)
        cost_matrix = self.cost._get_quadratic_cost_matrix(target_velocity_trajectory)
        # test shape
        assert cost_matrix.shape == torch.Size([10, 5, 2, 2])
        if target_velocity == [0.0, 0.0]:
            # make sure cost matrix is all 0 when target velocity is 0 in norm
            assert torch.allclose(cost_matrix, torch.zeros(10, 5, 2, 2))
        if target_velocity == [1.0, 1.0]:
            # make sure cost matrix represents ellipse with the principal axis given by velocity
            # direction
            orthogonal_matrix_test = torch.Tensor(
                [[1 / sqrt(2.0), -1 / sqrt(2.0)], [1 / sqrt(2.0), 1 / sqrt(2.0)]]
            ).expand(10, 5, -1, -1)
            cost_matrix_test = (
                orthogonal_matrix_test
                @ torch.Tensor(
                    [
                        [
                            [self.cost_params.scale_longitudinal, 0.0],
                            [0.0, self.cost_params.scale_lateral],
                        ]
                    ]
                )
                @ orthogonal_matrix_test.transpose(-1, -2)
            )
            assert torch.allclose(cost_matrix, cost_matrix_test)

    @pytest.mark.parametrize(
        "target_velocity, ego_position, reduce",
        [
            ([0.0, 0.0], [1.0, 1.0], "mean"),
            ([0.0, 0.0], [1.0, 1.0], "max"),
            ([0.0, 0.0], [1.0, -1.0], "mean"),
            ([0.0, 0.0], [1.0, -1.0], "max"),
            ([1.0, 1.0], [1.0, 1.0], "final"),
            ([1.0, 1.0], [1.0, 1.0], "max"),
            ([1.0, 1.0], [1.0, -1.0], "min"),
            ([1.0, 1.0], [1.0, -1.0], "mean"),
        ],
    )
    def test_call(self, target_velocity, ego_position, reduce):
        ego_position_trajectory = torch.Tensor(ego_position).expand(10, 5, -1)
        target_position_trajectory = torch.Tensor([0.0, 0.0]).expand(10, 5, -1)
        target_velocity_trajectory = torch.Tensor(target_velocity).expand(10, 5, -1)
        self.cost._reduce_fun_name = reduce
        cost = self.cost(
            ego_position_trajectory,
            target_position_trajectory,
            target_velocity_trajectory,
        )
        assert cost.shape == torch.Size([10])
        if target_velocity == [0.0, 0.0]:
            # make sure cost is zero when target velocity is 0 in norm
            assert torch.allclose(cost, torch.zeros(10))
        else:
            if ego_position == [1.0, 1.0]:
                # make sure cost is scaled with cost_scale_longitudinal when position error is
                # aligned with target velocity direction
                assert torch.allclose(
                    cost,
                    torch.Tensor(
                        [self.cost_params.scale_longitudinal * (1.0 ** 2 + 1.0 ** 2)]
                    ).expand(10),
                )
            if ego_position == [1.0, -1.0]:
                # make sure cost is scaled with cost_scale_lateral when position error is
                # perpendicular to target velocity direction
                assert torch.allclose(
                    cost,
                    torch.Tensor(
                        [self.cost_params.scale_lateral * (1.0 ** 2 + 1.0 ** 2)]
                    ).expand(10),
                )
