import atexit
import copy
import os
from mmcv import Config
import numpy as np
import pytest
from pytorch_lightning import seed_everything
import torch
import shutil
from torch.utils.data import DataLoader

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import SceneDataset, RandomSceneParams
from risk_biased.scene_dataset.scene import load_create_dataset


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


# atexit.register(clean_up_dataset_dir)


@pytest.fixture(scope="module")
def params():
    seed_everything(0)
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
    return cfg


@pytest.mark.parametrize(
    "n_data, batch_size, sample_times, state_dim",
    [(1024, 128, [0.0, 1.0, 2.0, 3.0, 4.0], 2)],
)
def test_load_data(params, n_data, batch_size, sample_times, state_dim):

    params = copy.deepcopy(params)
    params.batch_size = batch_size
    params.sample_times = sample_times
    scene_params = RandomSceneParams.from_config(params)

    dataset_rand = SceneDataset(n_data, scene_params, pre_fetch=False)
    data_loader_rand = DataLoader(
        dataset_rand, batch_size, collate_fn=dataset_rand.collate_fn, shuffle=False
    )

    dataset_prefetch = SceneDataset(n_data, scene_params, pre_fetch=True)
    data_loader_prefetch = DataLoader(
        dataset_prefetch,
        batch_size,
        collate_fn=dataset_prefetch.collate_fn,
        shuffle=False,
    )

    for i, (batch_rand, batch_prefetch) in enumerate(
        zip(data_loader_rand, data_loader_prefetch)
    ):
        if i == 0:
            first_batch_prefetch = batch_prefetch
            first_batch_rand = batch_rand
        # Check the shape of the data is the expected one
        assert (
            batch_rand.shape
            == batch_prefetch.shape
            == (batch_size, 1, len(sample_times), state_dim)
        )
    # Shuffle false and pre-fetch should loop back to the same batch
    assert torch.allclose(next(iter(data_loader_prefetch)), first_batch_prefetch)
    # Shuffle false but producing random batches should not loop back to the same batch
    assert not torch.allclose(next(iter(data_loader_rand)), first_batch_rand)


class TestDataset:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        clean_up_dataset_dir()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        [data_train, data_val, data_test] = load_create_dataset(params, current_dir)
        self.loaders = SceneDataLoaders(
            params.state_dim,
            params.num_steps,
            params.num_steps_future,
            params.batch_size,
            data_train=data_train,
            data_val=data_val,
            data_test=data_test,
            num_workers=params.num_workers,
        )
        assert os.path.exists(os.path.join(current_dir, "scene_dataset_000"))
        assert not os.path.exists(os.path.join(current_dir, "scene_dataset_001"))
        self.batch = torch.rand(
            params.batch_size,
            params.num_steps + params.num_steps_future,
            params.state_dim,
        )
        self.normalized_batch, self.offset = self.loaders.normalize_trajectory(
            self.batch
        )
        (
            self.normalized_batch_past,
            self.normalized_batch_future,
        ) = self.loaders.split_trajectory(self.normalized_batch)
        # Setup is done but some cleanup must be defined
        yield
        # Remove data directory after use
        dataset_dir = os.path.join(current_dir, "scene_dataset_000")
        shutil.rmtree(dataset_dir)

    def test_setup_datasets(self, params):
        current_dir = os.path.dirname(os.path.realpath(__file__))

        assert os.path.exists(os.path.join(current_dir, "scene_dataset_000"))
        # Should only load from directory that was created, not create a new one
        [train_set, val_set, test_set] = load_create_dataset(
            params, base_dir=current_dir
        )

        assert not os.path.exists(os.path.join(current_dir, "scene_dataset_001"))

        train_path = os.path.join(
            current_dir, "scene_dataset_000", "scene_dataset_train.npy"
        )
        val_path = os.path.join(
            current_dir, "scene_dataset_000", "scene_dataset_val.npy"
        )
        test_path = os.path.join(
            current_dir, "scene_dataset_000", "scene_dataset_test.npy"
        )

        # make sure paths for datasets exist
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(test_path)

        # make sure datasets match the specifications made in config
        assert np.load(train_path).shape == (
            2,
            params.datasets_sizes["train"],
            1,
            params.num_steps + params.num_steps_future,
            params.state_dim,
        )
        assert np.load(val_path).shape == (
            2,
            params.datasets_sizes["val"],
            1,
            params.num_steps + params.num_steps_future,
            params.state_dim,
        )
        assert np.load(test_path).shape == (
            2,
            params.datasets_sizes["test"],
            1,
            params.num_steps + params.num_steps_future,
            params.state_dim,
        )

        total_steps = params.num_steps + params.num_steps_future
        assert list(train_set.shape) == [
            2,
            params.datasets_sizes.train,
            1,
            total_steps,
            2,
        ]
        assert list(val_set.shape) == [2, params.datasets_sizes.val, 1, total_steps, 2]
        assert list(test_set.shape) == [
            2,
            params.datasets_sizes.test,
            1,
            total_steps,
            2,
        ]

    def test_split_trajectory(self, params):
        batch_history, batch_future = self.loaders.split_trajectory(self.batch)
        # make sure split_trajectory splits batch into history and future
        assert torch.all(torch.eq(batch_history, self.batch[:, : params.num_steps, :]))
        assert torch.all(torch.eq(batch_future, self.batch[:, params.num_steps :, :]))

    def test_normalize_trajectory(self, params):
        batch_copied = self.batch.detach().clone()
        # make sure batch remains the same
        assert torch.all(torch.eq(batch_copied, self.batch))
        # test normalization of whole batch
        assert torch.allclose(
            self.normalized_batch + self.offset.unsqueeze(1), self.batch
        )
        assert torch.allclose(
            self.batch - self.offset.unsqueeze(1), self.normalized_batch
        )

        batch_past, batch_fut = self.loaders.split_trajectory(self.batch)
        # test normalization of history
        assert torch.allclose(
            self.normalized_batch_past + self.offset.unsqueeze(1), batch_past
        )

    def test_unnormalize_trajectory(self, params):
        batch_future_test = self.loaders.unnormalize_trajectory(
            self.normalized_batch_future, self.offset
        )
        # test unnormalization
        assert torch.allclose(
            self.normalized_batch_future + self.offset.unsqueeze(1), batch_future_test
        )
