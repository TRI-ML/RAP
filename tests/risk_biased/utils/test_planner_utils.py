import math
import os

import pytest
import torch
from mmcv import Config

from risk_biased.mpc_planner.dynamics import PositionVelocityDoubleIntegrator
from risk_biased.mpc_planner.planner_cost import TrackingCost, TrackingCostParams
from risk_biased.utils.cost import TTCCostTorch, TTCCostParams
from risk_biased.utils.risk import get_risk_estimator
from risk_biased.utils.planner_utils import (
    to_state,
    get_interaction_cost,
    evaluate_risk,
    evaluate_control_sequence,
)


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    waymo_config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "waymo_config.py"
    )
    paths = [config_path, waymo_config_path]
    if isinstance(paths, str):
        cfg = Config.fromfile(paths)
    else:
        cfg = Config.fromfile(paths[0])
        for path in paths[1:]:
            c = Config.fromfile(path)
            cfg.update(c)

    cfg.num_control_samples = 10

    cfg.dt = 0.1
    cfg.num_steps = 3
    cfg.num_steps_future = 5
    cfg.state_dim = 5

    cfg.tracking_cost_scale_longitudinal = 0.1
    cfg.tracking_cost_scale_lateral = 1.0
    cfg.tracking_cost_reduce = "mean"

    cfg.cost_scale = 10
    cfg.cost_reduce = "mean"
    cfg.distance_bandwidth = 2
    cfg.time_bandwidth = 0.5
    cfg.min_velocity_diff = 0.01

    cfg.risk_estimator = {"type": "cvar", "eps": 1e-3}

    return cfg


class TestPlannerUtils:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        self.dynamics_model = PositionVelocityDoubleIntegrator(0.1)
        self.interaction_cost_function = TTCCostTorch(TTCCostParams.from_config(params))
        self.tracking_cost_function = TrackingCost(
            TrackingCostParams.from_config(params)
        )
        self.risk_estimator = get_risk_estimator(params.risk_estimator)
        self.dt = params.dt

    @pytest.mark.parametrize("ndim, sequence_size", [(2, 2), (3, 4), (4, 3)])
    def test_translate_position(self, ndim: int, sequence_size: int):
        translation_m = torch.Tensor([1.0, 2.0])

        state_pos = torch.zeros(sequence_size, 2)
        while state_pos.ndim < ndim:
            state_pos = state_pos.unsqueeze(0)
        state_pos = to_state(state_pos, self.dt)
        translated_state_pos = state_pos.translate(translation_m)
        assert torch.allclose(
            translated_state_pos.get_states(), state_pos.get_states() + translation_m
        )

        state_double_integrator = torch.zeros(sequence_size, 4)
        while state_double_integrator.ndim < ndim:
            state_double_integrator = state_double_integrator.unsqueeze(0)
        state_double_integrator = to_state(state_double_integrator, self.dt)
        translated_state_double_integrator = state_double_integrator.translate(
            translation_m
        )
        assert torch.allclose(
            translated_state_double_integrator.get_states(5),
            translated_state_pos.get_states(5),
        )

    @pytest.mark.parametrize("ndim, sequence_size", [(2, 3), (3, 5), (4, 2)])
    def test_rotate_angle(self, ndim: int, sequence_size: int):
        rotation_rad = torch.Tensor([math.pi / 2])

        state_pos = torch.ones(sequence_size, 2)
        while state_pos.ndim < ndim:
            state_pos = state_pos.unsqueeze(0)
        state_pos = to_state(state_pos, self.dt)
        rotated_state_pos = state_pos.rotate(rotation_rad)
        assert torch.allclose(
            rotated_state_pos.get_states(),
            torch.Tensor([-1.0, 1.0]).expand_as(state_pos.get_states()),
        )

        state_double_integrator = torch.Tensor([[1.0, 1.0, -1.0, 1.0]]).repeat(
            sequence_size, 1
        )
        while state_double_integrator.ndim < ndim:
            state_double_integrator = state_double_integrator.unsqueeze(0)
        state_double_integrator = to_state(state_double_integrator, self.dt)
        rotated_state_double_integrator = state_double_integrator.rotate(rotation_rad)
        assert torch.allclose(
            rotated_state_double_integrator.get_states(2),
            rotated_state_pos.get_states(2),
        )
        assert torch.allclose(
            rotated_state_double_integrator.get_states(4),
            torch.Tensor([-1.0, 1.0, -1.0, -1.0]).expand_as(
                rotated_state_pos.get_states(4)
            ),
        )

    @pytest.mark.parametrize(
        "with_ado_batch_dim, num_prediction_samples, num_agents",
        [(True, 1, 1), (False, 1, 2), (True, 5, 1), (False, 5, 2)],
    )
    def test_get_interaction_cost(
        self, params, with_ado_batch_dim, num_prediction_samples, num_agents
    ):
        ego_state_future = to_state(
            torch.randn(
                params.num_control_samples, 1, params.num_steps_future, params.state_dim
            ),
            params.dt,
        )

        if not with_ado_batch_dim:
            ado_position_future_samples = to_state(
                torch.randn(
                    num_prediction_samples,
                    num_agents,
                    params.num_steps_future,
                    params.state_dim,
                ),
                params.dt,
            )
        else:
            ado_position_future_samples = to_state(
                torch.randn(
                    num_prediction_samples,
                    num_agents,
                    params.num_steps_future,
                    params.state_dim,
                ),
                params.dt,
            )

        cost = get_interaction_cost(
            ego_state_future,
            ado_position_future_samples,
            self.interaction_cost_function,
        )

        assert cost.shape == torch.Size(
            [params.num_control_samples, num_agents, num_prediction_samples]
        )

    @pytest.mark.parametrize(
        "num_prediction_samples, num_agents, risk_level",
        [(1, 1, 0.0), (5, 2, 0.0), (1, 1, 0.9), (5, 2, 0.9)],
    )
    def test_evaluate_risk(
        self, params, num_prediction_samples, num_agents, risk_level
    ):
        cost = torch.rand(
            params.num_control_samples, num_agents, num_prediction_samples
        )
        weights = (
            torch.rand(params.num_control_samples, num_agents, num_prediction_samples)
            / num_prediction_samples
        )
        risk = evaluate_risk(risk_level, cost, weights, self.risk_estimator)
        if risk_level is None or risk_level == 0.0:
            assert torch.allclose(risk, cost.mean(dim=2))
        assert risk.shape == torch.Size([params.num_control_samples, num_agents])

    @pytest.mark.parametrize("risk_level", [(0.0), (0.5)])
    def test_evaluate_control_sequence(self, params, risk_level):
        num_prediction_samples = 5
        num_agents = 1

        control_sequence = torch.randn(
            1, params.num_steps_future, self.dynamics_model.control_dim
        )
        ego_state_history = to_state(
            torch.randn(1, params.num_steps, params.state_dim), self.dt
        )
        ego_state_target_trajectory = to_state(
            torch.randn(1, params.num_steps_future, params.state_dim), self.dt
        )

        ado_state_future_samples = to_state(
            torch.randn(num_prediction_samples, num_agents, params.num_steps_future, 2),
            params.dt,
        )
        weights = (
            torch.rand(num_prediction_samples, num_agents) / num_prediction_samples
        )
        interaction_risk, tracking_cost = evaluate_control_sequence(
            control_sequence,
            self.dynamics_model,
            ego_state_history,
            ego_state_target_trajectory,
            ado_state_future_samples,
            weights,
            self.interaction_cost_function,
            self.tracking_cost_function,
            risk_level,
            self.risk_estimator,
        )
        assert interaction_risk > 0.0
        assert tracking_cost > 0.0
