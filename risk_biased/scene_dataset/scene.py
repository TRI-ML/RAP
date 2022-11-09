from dataclasses import dataclass
import os
from typing import Union, List, Optional
import warnings
import copy

from mmcv import Config
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from risk_biased.scene_dataset.pedestrian import RandomPedestrians
from risk_biased.utils.torch_utils import torch_linspace


@dataclass
class RandomSceneParams:
    """Dataclass that defines all the listed parameters that are necessary for a RandomScene object

    Args:
        batch_size: number of scenes in the batch
        time_scene: time length of the scene in seconds
        sample_times: list of times to get the positions
        ego_ref_speed: constant reference speed of the ego vehicle in meters/seconds
        ego_speed_init_low: lowest initial speed of the ego vehicle in meters/seconds
        ego_speed_init_high: higest initial speed of the ego vehicle in meters/seconds
        ego_acceleration_mean_low: lowest mean acceleration of the ego vehicle in m/s^2
        ego_acceleration_mean_high: highest mean acceleration of the ego vehicle in m/s^2
        ego_acceleration_std: std for acceleration of the ego vehicle in m/s^2
        ego_length: length of the ego vehicle in meters
        ego_width: width of the ego vehicle in meters
        dt: time step to use in the trajectory sequence
        fast_speed: fast walking speed for the random pedestrian in meters/seconds
        slow_speed: slow walking speed for the random pedestrian in meters/seconds
        p_change_pace: probability that a slow (resp. fast) pedestrian walk at fast_speed (resp. slow_speed) at each time step
        proportion_fast: proportion of the pedestrians that are mainly walking at fast_speed
        perception_noise_std: standard deviation of the gaussian noise that is affecting the position observations
    """

    batch_size: int
    time_scene: float
    sample_times: list
    ego_ref_speed: float
    ego_speed_init_low: float
    ego_speed_init_high: float
    ego_acceleration_mean_low: float
    ego_acceleration_mean_high: float
    ego_acceleration_std: float
    ego_length: float
    ego_width: float
    dt: float
    fast_speed: float
    slow_speed: float
    p_change_pace: float
    proportion_fast: float
    perception_noise_std: float

    @staticmethod
    def from_config(cfg: Config):
        return RandomSceneParams(
            batch_size=cfg.batch_size,
            sample_times=cfg.sample_times,
            time_scene=cfg.time_scene,
            ego_ref_speed=cfg.ego_ref_speed,
            ego_speed_init_low=cfg.ego_speed_init_low,
            ego_speed_init_high=cfg.ego_speed_init_high,
            ego_acceleration_mean_low=cfg.ego_acceleration_mean_low,
            ego_acceleration_mean_high=cfg.ego_acceleration_mean_high,
            ego_acceleration_std=cfg.ego_acceleration_std,
            ego_length=cfg.ego_length,
            ego_width=cfg.ego_width,
            dt=cfg.dt,
            fast_speed=cfg.fast_speed,
            slow_speed=cfg.slow_speed,
            p_change_pace=cfg.p_change_pace,
            proportion_fast=cfg.proportion_fast,
            perception_noise_std=cfg.perception_noise_std,
        )


class RandomScene:
    """
    Batched scenes with one vehicle at constant velocity and one random pedestrian. Utility functions to draw the scene and compute risk factors (time to collision etc...)

    Args:
        params: dataclass containing the necessary parameters
        is_torch: set to True to produce Tensor batches and to False to produce numpy arrays
    """

    def __init__(
        self,
        params: RandomSceneParams,
        is_torch: bool = False,
    ) -> None:

        self._is_torch = is_torch
        self._batch_size = params.batch_size
        self._fast_speed = params.fast_speed
        self._slow_speed = params.slow_speed
        self._p_change_pace = params.p_change_pace
        self._proportion_fast = params.proportion_fast
        self.dt = params.dt
        self.sample_times = params.sample_times
        self.ego_ref_speed = params.ego_ref_speed
        self._ego_speed_init_low = params.ego_speed_init_low
        self._ego_speed_init_high = params.ego_speed_init_high
        self._ego_acceleration_mean_low = params.ego_acceleration_mean_low
        self._ego_acceleration_mean_high = params.ego_acceleration_mean_high
        self._ego_acceleration_std = params.ego_acceleration_std
        self.perception_noise_std = params.perception_noise_std
        self.road_length = (
            params.ego_ref_speed + params.fast_speed
        ) * params.time_scene
        self.time_scene = params.time_scene
        self.lane_width = 3
        self.sidewalks_width = 1.5
        self.road_width = 2 * self.lane_width + 2 * self.sidewalks_width
        self.bottom = -self.lane_width / 2 - self.sidewalks_width
        self.top = 3 * self.lane_width / 2 + self.sidewalks_width
        self.ego_width = 1.75
        self.ego_length = 4
        self.current_time = 0

        if self._is_torch:
            pedestrians_x = (
                torch.rand(params.batch_size, 1)
                * (self.road_length - self.ego_length / 2)
                + self.ego_length / 2
            )
            pedestrians_y = (
                torch.rand(params.batch_size, 1) * (self.top - self.bottom)
                + self.bottom
            )
            self._pedestrians_positions = torch.stack(
                (pedestrians_x, pedestrians_y), -1
            )
        else:
            pedestrians_x = np.random.uniform(
                low=self.ego_length / 2,
                high=self.road_length,
                size=(params.batch_size, 1),
            )
            pedestrians_y = np.random.uniform(
                low=self.bottom, high=self.top, size=(params.batch_size, 1)
            )
            self._pedestrians_positions = np.stack((pedestrians_x, pedestrians_y), -1)

        self.pedestrians = RandomPedestrians(
            batch_size=self._batch_size,
            dt=self.dt,
            fast_speed=self._fast_speed,
            slow_speed=self._slow_speed,
            p_change_pace=self._p_change_pace,
            proportion_fast=self._proportion_fast,
            is_torch=self._is_torch,
        )
        self._set_pedestrians()

    @property
    def pedestrians_positions(self):
        # relative_positions = self._pedestrians_positions/[[(self.road_length - self.ego_length / 2), (self.top - self.bottom)]] - [[self.ego_length / 2, self.bottom]]
        return self._pedestrians_positions

    def set_pedestrians_states(
        self,
        relative_pedestrians_positions: Union[torch.Tensor, np.ndarray],
        pedestrians_angles: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """Force pedestrian initial states

        Args:
            relative_pedestrians_positions: Relative positions in the scene as percentage distance from left to right and from bottom to top
            pedestrians_angles: Pedestrian heading angles in radiants
        """
        if self._is_torch:
            assert isinstance(relative_pedestrians_positions, torch.Tensor)
        else:
            assert isinstance(relative_pedestrians_positions, np.ndarray)

        self._batch_size = relative_pedestrians_positions.shape[0]
        if (0 > relative_pedestrians_positions).any() or (
            relative_pedestrians_positions > 1
        ).any():
            warnings.warn(
                "Some of the given pedestrian initial positions are outside of the road range"
            )
        center_y = (self.top - self.bottom) * relative_pedestrians_positions[
            :, :, 1
        ] + self.bottom
        center_x = (
            self.road_length - self.ego_length / 2
        ) * relative_pedestrians_positions[:, :, 0] + self.ego_length / 2
        if self._is_torch:
            pedestrians_positions = torch.stack([center_x, center_y], -1)
        else:
            pedestrians_positions = np.stack([center_x, center_y], -1)

        self.pedestrians = RandomPedestrians(
            batch_size=self._batch_size,
            dt=self.dt,
            fast_speed=self._fast_speed,
            slow_speed=self._slow_speed,
            p_change_pace=self._p_change_pace,
            proportion_fast=self._proportion_fast,
            is_torch=self._is_torch,
        )
        self._pedestrians_positions = pedestrians_positions
        if pedestrians_angles is not None:
            self.pedestrians.angle = pedestrians_angles
        self._set_pedestrians()

    def _set_pedestrians(self):
        self.pedestrians_trajectories = self.sample_pedestrians_trajectories(
            self.sample_times
        )

        self.final_pedestrians_positions = self.pedestrians_trajectories[:, :, -1]

    def get_ego_ref_trajectory(self, time_sequence: list):
        """
        Returns only one ego reference trajectory and not a batch because it is always the same.
        Args:
        time_sequence: the time points at which to get the positions
        """
        out = np.array([[[[t * self.ego_ref_speed, 0] for t in time_sequence]]])
        if self._is_torch:
            return torch.from_numpy(out.astype("float32"))
        else:
            return out

    def get_pedestrians_velocities(self):
        """
        Returns the batch of mean pedestrian velocities between their positions and their final positions.
        """
        return (self.final_pedestrians_positions - self._pedestrians_positions)[
            :, None
        ] / self.time_scene

    def get_ego_ref_velocity(self):
        """
        Returns the reference ego velocity.
        """
        if self._is_torch:
            return torch.from_numpy(
                np.array([[[[self.ego_ref_speed, 0]]]], dtype="float32")
            )
        else:
            return np.array([[[[self.ego_ref_speed, 0]]]])

    def get_ego_ref_position(self):
        """
        Returns the current reference ego position (at set time self.current_time)
        """
        if self._is_torch:
            return torch.from_numpy(
                np.array(
                    [[[[self.ego_ref_speed * self.current_time, 0]]]], dtype="float32"
                )
            )
        else:
            return np.array([[[[self.ego_ref_speed * self.current_time, 0]]]])

    def set_current_time(self, time: float):
        """
        Set the current time of the scene.
        Args:
        time : The current time to set. It should be between 0 and 1
        """
        assert 0 <= time <= self.time_scene
        self.current_time = time

    def sample_ego_velocities(self, time_sequence: list):
        """
        Get ego velocity trajectories following the ego's acceleration distribution and the initial
        velocity distribution.

        Args:
            time_sequence: a list of time points at which to sample the trajectory positions.
        Returns:
            batch of sequence of velocities of shape (batch_size, 1, len(time_sequence), 2)
        """
        vel_traj = []
        # uniform sampling of acceleration_mean between self._ego_acceleration_mean_low and
        # self._ego_acceleration_mean_high
        acceleration_mean = np.random.rand(self._batch_size, 2) * np.array(
            [
                self._ego_acceleration_mean_high - self._ego_acceleration_mean_low,
                0.0,
            ]
        ) + np.array([self._ego_acceleration_mean_low, 0.0])
        t_prev = 0
        # uniform sampling of initial velocity between self._ego_speed_init_low and
        # self._ego_speed_init_high
        vel_prev = np.random.rand(self._batch_size, 2) * np.array(
            [self._ego_speed_init_high - self._ego_speed_init_low, 0.0]
        ) + np.array([self._ego_speed_init_low, 0.0])
        for t in time_sequence:
            # integrate accelerations once to get velocities
            acceleration = acceleration_mean + np.random.randn(
                self._batch_size, 2
            ) * np.array([self._ego_acceleration_std, 0.0])
            vel_prev = vel_prev + acceleration * (t - t_prev)
            t_prev = t
            vel_traj.append(vel_prev)
        vel_traj = np.stack(vel_traj, 1)
        if self._is_torch:
            vel_traj = torch.from_numpy(vel_traj.astype("float32"))
        return vel_traj[:, None]

    def sample_ego_trajectories(self, time_sequence: list):
        """
        Get ego trajectories following the ego's acceleration distribution and the initial velocity
        distribution.

        Args:
            time_sequence: a list of time points at which to sample the trajectory positions.
        Returns:
            batch of sequence of positions of shape (batch_size, len(time_sequence), 2)
        """
        vel_traj = self.sample_ego_velocities(time_sequence)
        traj = []
        t_prev = 0
        pos_prev = np.array([[0, 0]], dtype="float32")
        if self._is_torch:
            pos_prev = torch.from_numpy(pos_prev)
        for idx, t in enumerate(time_sequence):
            # integrate velocities once to get positions
            vel = vel_traj[:, :, idx, :]
            pos_prev = pos_prev + vel * (t - t_prev)
            t_prev = t
            traj.append(pos_prev)
        if self._is_torch:
            return torch.stack(traj, -2)
        else:
            return np.stack(traj, -2)

    def sample_pedestrians_trajectories(self, time_sequence: list):
        """
        Produce pedestrian trajectories following the pedestrian behavior distribution
        (it is resampled, the final position will not match self.final_pedestrians_positions)
        Args:
            time_sequence: a list of time points at which to sample the trajectory positions.
        Returns:
            batch of sequence of positions of shape (batch_size, len(time_sequence), 2)
        """
        traj = []
        t_prev = 0
        pos_prev = self.pedestrians_positions
        for t in time_sequence:
            pos_prev = (
                pos_prev
                + self.pedestrians.get_final_position(t - t_prev)
                - self.pedestrians.position
            )
            t_prev = t
            traj.append(pos_prev)
        if self._is_torch:
            traj = torch.stack(traj, 2)
            return traj + torch.randn_like(traj) * self.perception_noise_std
        else:
            traj = np.stack(traj, 2)
            return traj + np.random.randn(*traj.shape) * self.perception_noise_std

    def get_pedestrians_trajectories(self):
        """
        Returns the batch of pedestrian trajectories sampled every dt.
        """
        return self.pedestrians_trajectories

    def get_pedestrian_trajectory(self, ind: int, time_sequence: list = None):
        """
        Returns one pedestrian trajectory of index ind sampled at times set in time_sequence.
        Args:
            ind: index of the pedestrian in the batch.
            time_sequence: a list of time points at which to sample the trajectory positions.
        Returns:
            A pedestrian trajectory of shape (len(time_sequence), 2)
        """
        len_traj = len(self.sample_times)
        if self._is_torch:
            ped_traj = torch_linspace(
                self.pedestrians_positions[ind],
                self.final_pedestrians_positions[ind],
                len_traj,
            )
        else:
            ped_traj = np.linspace(
                self.pedestrians_positions[ind],
                self.final_pedestrians_positions[ind],
                len_traj,
            )

        if time_sequence is not None:
            n_steps = [int(t / self.dt) for t in time_sequence]
        else:
            n_steps = range(int(self.time_scene / self.dt))
        return ped_traj[n_steps]


class SceneDataset(Dataset):
    """
    Dataset of scenes with one vehicle at constant velocity and one random pedestrian.
    The scenes are randomly generated so the distribution can be sampled at each batch or pre-fetched.

    Args:
        len: int number of scenes per epoch
        params: dataclass defining all the necessary parameters
        pre_fetch: set to True to fetch the whole dataset at initialization
    """

    def __init__(
        self,
        len: int,
        params: RandomSceneParams,
        pre_fetch: bool = True,
    ) -> None:
        super().__init__()
        self._pre_fetch = pre_fetch
        self._len = len
        self._sample_times = params.sample_times
        self.params = copy.deepcopy(params)
        params.batch_size = len
        if self._pre_fetch:
            self.scene_set = RandomScene(
                params, is_torch=True
            ).sample_pedestrians_trajectories(self._sample_times)

    def __len__(self) -> int:
        return self._len

    # This is a hack, get item only returns the index so that the collate_fn can handle making the batch without looping on RandomScene creation.
    def __getitem__(self, index: int) -> Tensor:
        return index

    def collate_fn(self, index_list: list) -> Tensor:
        if self._pre_fetch:
            return self.scene_set[torch.from_numpy(np.array(index_list))]
        else:
            self.params.batch_size = len(index_list)
            return RandomScene(
                self.params,
                is_torch=True,
            ).sample_pedestrians_trajectories(self._sample_times)


# Call this function to create a dataset as a .npy file that can be loaded as a numpy array with np.load(file_name.npy)
def save_dataset(file_path: str, size: int, config: Config):
    """
    Save a dataset at file_path using the configuration.
    Args:
        file_path: Where to save the dataset
        size: Number of samples to save
        config: Configuration to use for the dataset generation
    """
    dir_path = os.path.dirname(file_path)
    config_path = os.path.join(dir_path, "config.py")
    config = copy.deepcopy(config)
    config.batch_size = size
    params = RandomSceneParams.from_config(config)
    scene = RandomScene(
        params,
        is_torch=False,
    )
    data_pedestrian = scene.sample_pedestrians_trajectories(config.sample_times)
    data_ego = scene.sample_ego_trajectories(config.sample_times)
    data = np.stack([data_pedestrian, data_ego], 0)
    np.save(file_path, data)
    # Cannot use config.dump here because it is buggy and does not work if config was not loaded from a file.
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.pretty_text)


def load_create_dataset(
    config: Config,
    base_dir=None,
) -> List:
    """
    Load the dataset described by its config if it exists or create one.

    Args:
        config: Configuration to use for the dataset
        base_dir: Where to look for the dataset or to save it.
    """

    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "data"
        )
    found = False
    dataset_out = []
    i = 0
    dir_path = os.path.join(base_dir, f"scene_dataset_{i:03d}")
    while os.path.exists(dir_path):
        config_path = os.path.join(dir_path, "config.py")
        if os.path.exists(config_path):
            config_check = Config.fromfile(config_path)
            if config_check.dataset_parameters == config.dataset_parameters:
                found = True
                break
        else:
            warnings.warn(
                f"Dataset directory {dir_path} exists but doesn't contain a config file. Cannot use it."
            )
        i += 1
        dir_path = os.path.join(base_dir, f"scene_dataset_{i:03d}")

    if not found:
        print(f"Dataset not found, creating a new one.")
        os.makedirs(dir_path)
        for dataset in config.datasets:
            dataset_name = f"scene_dataset_{dataset}.npy"
            dataset_path = os.path.join(dir_path, dataset_name)
            save_dataset(dataset_path, config.datasets_sizes[dataset], config)
    if found:
        print(f"Loading existing dataset at {dir_path}.")

    for dataset in config.datasets:
        dataset_path = os.path.join(dir_path, f"scene_dataset_{dataset}.npy")
        dataset_out.append(torch.from_numpy(np.load(dataset_path).astype("float32")))

    return dataset_out
