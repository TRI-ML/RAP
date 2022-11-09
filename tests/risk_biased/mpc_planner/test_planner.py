import os
import pytest
import torch
from mmcv import Config

from risk_biased.mpc_planner.planner import MPCPlanner, MPCPlannerParams
from risk_biased.predictors.biased_predictor import (
    LitTrajectoryPredictorParams,
    LitTrajectoryPredictor,
)

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.utils.cost import TTCCostParams
from risk_biased.utils.planner_utils import to_state


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = Config.fromfile(config_path)
    cfg.num_control_samples = 10
    cfg.num_elite = 3
    cfg.iter_max = 3
    cfg.smoothing_factor = 0.2
    cfg.mean_warm_start = True

    cfg.acceleration_std_x_m_s2 = 2.0
    cfg.acceleration_std_y_m_s2 = 0.0

    cfg.dt = 0.1
    cfg.num_steps = 3
    cfg.num_steps_future = 5

    cfg.tracking_cost_scale_longitudinal = 0.1
    cfg.tracking_cost_scale_lateral = 1.0
    cfg.tracking_cost_reduce = "mean"

    cfg.cost_scale = 10
    cfg.cost_reduce = "mean"
    cfg.distance_bandwidth = 2
    cfg.time_bandwidth = 0.5
    cfg.min_velocity_diff = 0.01

    cfg.risk_estimator = {"type": "cvar", "eps": 1e-3}

    cfg.interaction_type = ""
    cfg.mcg_dim_expansion = 2
    cfg.mcg_num_layers = 0
    cfg.num_attention_heads = 4
    cfg.num_blocks = 3
    cfg.sequence_encoder_type = "MLP"  # one of "MLP", "LSTM", "maskedLSTM"
    cfg.sequence_decoder_type = "MLP"  # one of "MLP", "LSTM"

    cfg.state_dim = 2
    cfg.dynamic_state_dim = 2
    cfg.map_state_dim = 2
    cfg.max_size_lane = 0
    cfg.latent_dim = 2
    cfg.hidden_dim = 64
    cfg.num_hidden_layers = 3
    cfg.risk_distribution = {"type": "log-uniform", "min": 0, "max": 1, "scale": 3}
    cfg.kl_weight = 1.0
    cfg.kl_threshold = 0.1
    cfg.learning_rate = 1e-3
    cfg.n_mc_samples_risk = 2048
    cfg.n_mc_samples_biased = 128
    cfg.risk_weight = 1e3
    cfg.use_risk_constraint = True
    cfg.risk_constraint_update_every_n_epoch = 20
    cfg.risk_constraint_weight_update_factor = 1.5
    cfg.risk_constraint_weight_maximum = 1e5
    cfg.condition_on_ego_future = True
    cfg.is_mlp_residual = True
    cfg.num_samples_min_fde = 6

    return cfg


class TestMPCPlanner:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        self.planner_params = MPCPlannerParams.from_config(params)
        predictor_params = LitTrajectoryPredictorParams.from_config(params)
        self.predictor = LitTrajectoryPredictor(
            predictor_params,
            TTCCostParams.from_config(params),
            SceneDataLoaders.unnormalize_trajectory,
        )
        self.normalizer = SceneDataLoaders.normalize_trajectory
        self.planner = MPCPlanner(self.planner_params, self.predictor, self.normalizer)

    def test_reset(self):
        self.planner.reset()
        assert torch.allclose(
            self.planner.solver.control_input_mean_init,
            self.planner.control_input_mean_init,
        )
        assert torch.allclose(
            self.planner.solver.control_input_std_init,
            self.planner.control_input_std_init,
        )
        assert self.planner._ego_state_history == []
        assert self.planner._ego_state_target_trajectory == None
        assert self.planner._ego_state_planned_trajectory == None

        assert self.planner._ado_state_history == []
        assert self.planner._latest_ado_position_future_samples == None

    def test_replan(self, params):
        num_prediction_samples = 100
        num_agents = 1
        self.planner.reset()
        current_ego_state = to_state(torch.Tensor([[1, 1, 0, 0]]), params.dt)
        for step in range(params.num_steps + 1):
            self.planner._update_ego_state_history(current_ego_state)

        current_ado_state = to_state(torch.Tensor([[2.0, 0.0, 0, 0]]), params.dt)
        for step in range(params.num_steps + 1):
            self.planner._update_ado_state_history(current_ado_state)

        target_velocity = torch.Tensor([3.0, 0.0])

        self.planner.replan(
            current_ado_state,
            current_ego_state,
            target_velocity,
            num_prediction_samples=num_prediction_samples,
        )
        assert self.planner._ego_state_planned_trajectory.shape == torch.Size(
            [num_agents, params.num_steps_future]
        )
        next_ego_state = self.planner.get_planned_next_ego_state()
        assert next_ego_state.shape == torch.Size([1])
        assert self.planner.fetch_latest_prediction().shape == torch.Size(
            [num_prediction_samples, num_agents, params.num_steps_future]
        )
