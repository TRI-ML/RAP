import os
import pytest
import torch
from mmcv import Config

from risk_biased.mpc_planner.dynamics import PositionVelocityDoubleIntegrator
from risk_biased.mpc_planner.planner_cost import TrackingCost, TrackingCostParams
from risk_biased.mpc_planner.solver import CrossEntropySolver, CrossEntropySolverParams
from risk_biased.predictors.biased_predictor import (
    LitTrajectoryPredictorParams,
    LitTrajectoryPredictor,
)

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.utils.cost import TTCCostTorch, TTCCostParams
from risk_biased.utils.risk import get_risk_estimator
from risk_biased.utils.planner_utils import to_state


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    planning_config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "planning_config.py"
    )
    paths = [config_path, planning_config_path]
    if isinstance(paths, str):
        cfg = Config.fromfile(paths)
    else:
        cfg = Config.fromfile(paths[0])
        for path in paths[1:]:
            c = Config.fromfile(path)
            cfg.update(c)
    cfg.num_control_samples = 10
    cfg.num_elite = 3
    cfg.iter_max = 3
    cfg.smoothing_factor = 0.2
    cfg.mean_warm_start = True

    cfg.num_steps = 3
    cfg.num_steps_future = 5

    cfg.state_dim = 5
    cfg.dynamic_state_dim = 5
    cfg.map_state_dim = 2
    cfg.max_size_lane = 2
    cfg.latent_dim = 2
    cfg.hidden_dim = 64
    cfg.num_hidden_layers = 3

    return cfg


class TestCrossEntropySolver:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        self.solver_params = CrossEntropySolverParams.from_config(params)
        self.dynamics_model = PositionVelocityDoubleIntegrator(params.dt)
        self.interaction_cost_function = TTCCostTorch(TTCCostParams.from_config(params))
        self.tracking_cost_function = TrackingCost(
            TrackingCostParams.from_config(params)
        )
        self.risk_estimator = get_risk_estimator(params.risk_estimator)

        self.control_input_mean_default = torch.randn(
            1, params.num_steps_future, self.dynamics_model.control_dim
        )
        self.control_input_std_default = torch.rand_like(
            self.control_input_mean_default
        )
        self.solver_default = CrossEntropySolver(
            self.solver_params,
            self.dynamics_model,
            self.control_input_mean_default,
            self.control_input_std_default,
            self.tracking_cost_function,
            self.interaction_cost_function,
            self.risk_estimator,
        )
        predictor_params = LitTrajectoryPredictorParams.from_config(params)
        self.predictor = LitTrajectoryPredictor(
            predictor_params,
            TTCCostParams.from_config(params),
            SceneDataLoaders.unnormalize_trajectory,
        )
        self.normalizer = SceneDataLoaders.normalize_trajectory

    def test_reset(self):
        self.solver_default.reset()
        assert self.solver_default._iter_current == 0
        assert torch.allclose(
            self.solver_default._control_input_mean, self.control_input_mean_default
        )
        assert torch.allclose(
            self.solver_default._control_input_std, self.control_input_std_default
        )
        assert self.solver_default._latest_ado_position_future_samples == None

    def test_get_elites(self, params):
        control_input = torch.randn(
            params.num_control_samples,
            1,
            params.num_steps_future,
            self.dynamics_model.control_dim,
        )
        risk = torch.Tensor(
            [0.0, 1.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
        ).unsqueeze(-1)
        elite_control_input, elite_risk = self.solver_default._get_elites(
            control_input, risk
        )
        assert elite_control_input.shape == torch.Size(
            [
                params.num_elite,
                1,
                params.num_steps_future,
                self.dynamics_model.control_dim,
            ]
        )
        assert elite_risk.shape == torch.Size([params.num_elite, 1])
        assert torch.allclose(elite_control_input, control_input[[0, 2, 4]])
        assert torch.allclose(elite_risk, torch.Tensor([0.0, 0.1, 0.2]).unsqueeze(-1))

    @pytest.mark.parametrize(
        "num_elite, smoothing_factor", ([3, 0.0], [3, 1.0], [1, 0.0], [1, 1.0])
    )
    def test_update_control_distribution(self, params, num_elite, smoothing_factor):
        solver = CrossEntropySolver(
            self.solver_params,
            self.dynamics_model,
            self.control_input_mean_default,
            self.control_input_std_default,
            self.tracking_cost_function,
            self.interaction_cost_function,
            self.risk_estimator,
        )
        solver.params.num_elite = num_elite
        solver.params.smoothing_factor = smoothing_factor
        elite_control_input = torch.ones(
            num_elite, params.num_steps_future, self.dynamics_model.control_dim
        )
        solver._update_control_distribution(elite_control_input)
        if smoothing_factor == 0.0:
            assert torch.allclose(
                solver._control_input_mean, torch.ones_like(solver._control_input_mean)
            )
            assert torch.allclose(
                solver._control_input_std, torch.zeros_like(solver._control_input_std)
            )
        else:
            assert torch.allclose(
                solver._control_input_mean, solver.control_input_mean_init
            )
            assert torch.allclose(
                solver._control_input_std, solver.control_input_std_init
            )

    @pytest.mark.parametrize(
        "risk_level, num_prediction_samples",
        [(0.0, 1), (0.0, 10), (0.5, 1), (0.5, 10)],
    )
    def test_sample_prediction(self, params, risk_level, num_prediction_samples):
        num_agents = 1
        ado_state_history = to_state(
            torch.randn(num_agents, params.num_steps, params.state_dim), params.dt
        )
        ego_state_history = to_state(
            torch.randn(1, params.num_steps, params.state_dim), params.dt
        )
        ego_state_future = to_state(
            torch.randn(1, params.num_steps_future, params.state_dim), params.dt
        )

        ado_position_future_samples, weights = CrossEntropySolver.sample_prediction(
            self.predictor,
            ado_state_history,
            self.normalizer,
            ego_state_history,
            ego_state_future,
            num_prediction_samples=num_prediction_samples,
            risk_level=risk_level,
        )
        assert ado_position_future_samples.shape == torch.Size(
            [num_prediction_samples, num_agents, params.num_steps_future]
        )

    @pytest.mark.parametrize(
        "mean_warm_start, risk_level, resample_prediction, risk_in_predictor",
        [
            (False, 0.0, False, False),
            (False, 0.0, False, True),
            (False, 0.0, True, False),
            (False, 0.0, True, True),
            (False, 0.5, False, False),
            (False, 0.5, False, True),
            (False, 0.5, True, False),
            (False, 0.5, True, True),
            (True, 0.0, False, False),
            (True, 0.0, False, True),
            (True, 0.0, True, False),
            (True, 0.0, True, True),
            (True, 0.5, False, False),
            (True, 0.5, False, True),
            (True, 0.5, True, False),
            (True, 0.5, True, True),
        ],
    )
    def test_solve(
        self,
        params,
        mean_warm_start,
        risk_level,
        resample_prediction,
        risk_in_predictor,
    ):
        num_prediction_samples = 5
        num_agents = 1

        ego_state_history = to_state(
            torch.randn(num_agents, params.num_steps, params.state_dim), params.dt
        )
        ego_state_target_trajectory = to_state(
            torch.randn(num_agents, params.num_steps_future, params.state_dim),
            params.dt,
        )

        ado_state_history = to_state(
            torch.randn(num_agents, params.num_steps, 2), params.dt
        )
        self.solver_default.params.mean_warm_start = mean_warm_start
        self.solver_default.solve(
            self.predictor,
            ego_state_history,
            ego_state_target_trajectory,
            ado_state_history,
            self.normalizer,
            num_prediction_samples=num_prediction_samples,
            risk_level=risk_level,
            resample_prediction=resample_prediction,
            risk_in_predictor=risk_in_predictor,
        )
        assert self.solver_default._iter_current == params.iter_max
        assert self.solver_default.fetch_latest_prediction().shape == torch.Size(
            [num_prediction_samples, num_agents, params.num_steps_future]
        )
        if not mean_warm_start:
            assert torch.allclose(
                self.solver_default.control_input_mean_init,
                self.control_input_mean_default,
            )
            assert torch.allclose(
                self.solver_default.control_input_std_init,
                self.control_input_std_default,
            )
        else:
            assert torch.allclose(
                self.solver_default.control_input_mean_init[:, -1],
                self.control_input_mean_default[:, -1],
            )
            assert torch.allclose(
                self.solver_default.control_input_mean_init[:, :-1],
                self.solver_default._control_input_mean[:, 1:],
            )
            assert torch.allclose(
                self.solver_default.control_input_std_init,
                self.control_input_std_default,
            )
