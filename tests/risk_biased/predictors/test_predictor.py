import atexit

from mmcv import Config
import os
import pytest
from pytorch_lightning import seed_everything
import shutil
import torch

from risk_biased.scene_dataset.scene import load_create_dataset
from risk_biased.predictors.biased_predictor import (
    LitTrajectoryPredictor,
    LitTrajectoryPredictorParams,
)
from risk_biased.utils.cost import TTCCostParams
from risk_biased.scene_dataset.loaders import SceneDataLoaders


def clean_up_dataset_dir():
    """
    This function is designed to delete the directories
    that might have created even if the test fails early
    by being called on exit.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir0 = os.path.join(current_dir, "scene_dataset_000")
    if os.path.exists(dataset_dir0):
        shutil.rmtree(dataset_dir0)
    dataset_dir1 = os.path.join(current_dir, "scene_dataset_001")
    if os.path.exists(dataset_dir1):
        shutil.rmtree(dataset_dir1)


atexit.register(clean_up_dataset_dir)


@pytest.fixture(scope="module")
def params():
    seed_everything(0)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cfg = Config()
    cfg.batch_size = 4
    cfg.time_scene = 5.0
    cfg.dt = 0.1
    cfg.sample_times = [t * cfg.dt for t in range(0, int(cfg.time_scene / cfg.dt))]
    cfg.ego_ref_speed = 14
    cfg.ego_speed_init_low = 4.0
    cfg.ego_speed_init_high = 16.0
    cfg.ego_acceleration_mean_low = -1.5
    cfg.ego_acceleration_mean_high = 1.5
    cfg.ego_acceleration_std = 1.5
    cfg.ego_length = 4
    cfg.ego_width = 1.75
    cfg.fast_speed = 2.0
    cfg.slow_speed = 1.0
    cfg.p_change_pace = 0.2
    cfg.proportion_fast = 0.5
    cfg.perception_noise_std = 0.03
    cfg.state_dim = 2
    cfg.num_steps = 3
    cfg.num_steps_future = len(cfg.sample_times) - cfg.num_steps
    cfg.file_name = "test_scene_data"
    cfg.datasets_sizes = {"train": 100, "val": 10, "test": 30}
    cfg.datasets = list(cfg.datasets_sizes.keys())
    cfg.num_workers = 2
    cfg.dataset_parameters = {
        "dt": cfg.dt,
        "time_scene": cfg.time_scene,
        "sample_times": cfg.sample_times,
        "ego_ref_speed": cfg.ego_ref_speed,
        "ego_speed_init_low": cfg.ego_speed_init_low,
        "ego_speed_init_high": cfg.ego_speed_init_high,
        "ego_acceleration_mean_low": cfg.ego_acceleration_mean_low,
        "ego_acceleration_mean_high": cfg.ego_acceleration_mean_high,
        "ego_acceleration_std": cfg.ego_acceleration_std,
        "fast_speed": cfg.fast_speed,
        "slow_speed": cfg.slow_speed,
        "p_change_pace": cfg.p_change_pace,
        "proportion_fast": cfg.proportion_fast,
        "file_name": cfg.file_name,
        "datasets_sizes": cfg.datasets_sizes,
        "state_dim": cfg.state_dim,
        "num_steps": cfg.num_steps,
        "num_steps_future": cfg.num_steps_future,
        "perception_noise_std": cfg.perception_noise_std,
    }
    [data_train, data_val, data_test] = load_create_dataset(cfg, current_dir)
    loaders = SceneDataLoaders(
        cfg.state_dim,
        cfg.num_steps,
        cfg.num_steps_future,
        cfg.batch_size,
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        num_workers=cfg.num_workers,
    )
    return cfg, loaders


class TestPredictor:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        cfg, loaders = params
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Should create directory and datasets
        [train_set, val_set, test_set] = load_create_dataset(cfg, base_dir=current_dir)
        params = LitTrajectoryPredictorParams.from_config(cfg)
        cost_params = TTCCostParams.from_config(cfg)
        self.predictor = LitTrajectoryPredictor(
            params, cost_params, loaders.unnormalize_trajectory
        )
        assert not os.path.exists(os.path.join(current_dir, "scene_dataset_001"))
        self.batch = torch.rand(
            cfg.batch_size,
            1,
            cfg.num_steps + cfg.num_steps_future,
            cfg.state_dim,
        )
        self.normalized_batch, self.offset = loaders.normalize_trajectory(self.batch)
        (
            self.normalized_batch_past,
            self.normalized_batch_future,
        ) = loaders.split_trajectory(self.normalized_batch)

        # Remove after use
        dataset_dir = os.path.join(current_dir, "scene_dataset_000")
        shutil.rmtree(dataset_dir)
