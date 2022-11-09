from typing import Tuple, List
from cv2 import repeat
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import Tensor
import numpy as np
import pickle
import os

from mmcv import Config


class WaymoDataset(Dataset):
    """
    Dataset loader for custom preprocessed files of Waymo data.
    Args:
        path: path to the dataset directory
        args: global settings
    """

    def __init__(self, cfg: Config, split: str, input_angle: bool = True):
        super(WaymoDataset, self).__init__()
        self.p_exchange_two_first = 1
        if "val" in split.lower():
            path = cfg.val_dataset_path
        elif "test" in split.lower():
            path = cfg.test_dataset_path
        elif "sample" in split.lower():
            path = cfg.sample_dataset_path
        else:
            path = cfg.train_dataset_path
            self.p_exchange_two_first = cfg.p_exchange_two_first

        self.file_list = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isfile(os.path.join(path, name))
        ]
        self.normalize = cfg.normalize_angle
        # self.load_dataset(path, 16)
        # self.idx_list = list(self.dataset.keys())
        self.input_angle = input_angle
        self.hist_len = cfg.num_steps
        self.fut_len = cfg.num_steps_future
        self.time_len = self.hist_len + self.fut_len
        self.min_num_obs = cfg.min_num_observation
        self.max_size_lane = cfg.max_size_lane
        self.random_rotation = cfg.random_rotation
        self.random_translation = cfg.random_translation
        self.angle_std = cfg.angle_std
        self.translation_distance_std = cfg.translation_distance_std
        self.max_num_agents = cfg.max_num_agents
        self.max_num_objects = cfg.max_num_objects
        self.state_dim = cfg.state_dim
        self.map_state_dim = cfg.map_state_dim
        self.dt = cfg.dt

        if "val" in os.path.basename(path).lower():
            self.dataset_size_limit = cfg.val_dataset_size_limit
        else:
            self.dataset_size_limit = cfg.train_dataset_size_limit

    def __len__(self):
        if self.dataset_size_limit is not None:
            return min(len(self.file_list), self.dataset_size_limit)
        else:
            return len(self.file_list)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the item at index idx in the dataset. Normalize the scene and output absolute angle and position.
        Returns:
            trajectories, mask, mask_loss, lanes, mask_lanes, angle, mean_position
        """
        selected_file = self.file_list[idx]
        with open(selected_file, "rb") as handle:
            dataset = pickle.load(handle)
        rel_state_all = dataset["traj"]
        mask_all = dataset["mask_traj"]
        mask_loss = dataset["mask_to_predict"]
        rel_lane_all = dataset["lanes"]
        mask_lane_all = dataset["mask_lanes"]
        mean_pos = dataset["mean_pos"]
        assert (
            (
                rel_state_all[self.hist_len + 5 :, :, :2][mask_all[self.hist_len + 5 :]]
                != 0
            )
            .any(-1)
            .all()
        )
        assert (
            (
                rel_state_all[self.hist_len + 5 :, :, :2][
                    mask_loss[self.hist_len + 5 :]
                ]
                != 0
            )
            .any(-1)
            .all()
        )
        if "lane_states" in dataset.keys():
            lane_states = dataset["lane_states"]
        else:
            lane_states = None
        if np.random.rand() > self.p_exchange_two_first:
            rel_state_all[:, [0, 1]] = rel_state_all[:, [1, 0]]
            mask_all[:, [0, 1]] = mask_all[:, [1, 0]]
            mask_loss[:, [0, 1]] = mask_loss[:, [1, 0]]
        assert (
            (
                rel_state_all[self.hist_len + 5 :, :, :2][mask_all[self.hist_len + 5 :]]
                != 0
            )
            .any(-1)
            .all()
        )
        assert (
            (
                rel_state_all[self.hist_len + 5 :, :, :2][
                    mask_loss[self.hist_len + 5 :]
                ]
                != 0
            )
            .any(-1)
            .all()
        )
        if self.normalize:
            angle = rel_state_all[self.hist_len - 1, 1, 2]

            if self.random_rotation:
                if self.normalize:
                    angle += np.random.normal(0, self.angle_std)
                else:
                    angle += np.random.uniform(-np.pi, np.pi)
            if self.random_translation:
                distance = (
                    np.random.normal([0, 0], self.translation_distance_std, 2)
                    * mask_all[self.hist_len - 1 : self.hist_len, :, None]
                    - rel_state_all[self.hist_len - 1 : self.hist_len, 1:2, :2]
                )
            else:
                distance = -rel_state_all[self.hist_len - 1 : self.hist_len, 1:2, :2]

            rel_state_all[:, :, :2] += distance
            rel_lane_all[:, :, :2] += distance
            mean_pos += distance[0, 0, :]
            rel_state_all = self.scene_rotation(rel_state_all, -angle)
            rel_lane_all = self.scene_rotation(rel_lane_all, -angle)

        else:
            if self.random_translation:
                distance = np.random.normal([0, 0], self.translation_distance_std, 2)
                rel_state_all = (
                    rel_state_all
                    + mask_all[self.hist_len - 1 : self.hist_len, :, None] * distance
                )
                rel_lane_all = (
                    rel_lane_all
                    + mask_all[self.hist_len - 1 : self.hist_len, :, None] * distance
                )

            if self.random_rotation:
                angle = np.random.uniform(0, 2 * np.pi)
                rel_state_all = self.scene_rotation(rel_state_all, angle)
                rel_lane_all = self.scene_rotation(rel_lane_all, angle)
            else:
                angle = 0
        return (
            rel_state_all,
            mask_all,
            mask_loss,
            rel_lane_all,
            mask_lane_all,
            lane_states,
            angle,
            mean_pos,
            idx,
        )

    @staticmethod
    def scene_rotation(coor: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate all the coordinates with the same angle
        Args:
            coor: array of x, y coordinates
            angle: radiants to rotate the coordinates by
        Returns:
            coor_rotated
        """
        rot_matrix = np.zeros((2, 2))
        c = np.cos(angle)
        s = np.sin(angle)
        rot_matrix[0, 0] = c
        rot_matrix[0, 1] = -s
        rot_matrix[1, 0] = s
        rot_matrix[1, 1] = c
        coor[..., :2] = np.matmul(
            rot_matrix, np.expand_dims(coor[..., :2], axis=-1)
        ).squeeze(-1)
        if coor.shape[-1] > 2:
            coor[..., 2] += angle
        if coor.shape[-1] >= 5:
            coor[..., 3:5] = np.matmul(
                rot_matrix, np.expand_dims(coor[..., 3:5], axis=-1)
            ).squeeze(-1)
        return coor

    def fill_past(self, past, mask_past):
        current_velocity = past[..., 0, 3:5]
        for t in range(1, past.shape[-2]):
            current_velocity = torch.where(
                mask_past[..., t, None], past[..., t, 3:5], current_velocity
            )
            past[..., t, 3:5] = current_velocity
            predicted_position = past[..., t - 1, :2] + current_velocity * self.dt
            past[..., t, :2] = torch.where(
                mask_past[..., t, None], past[..., t, :2], predicted_position
            )
        return past

    def collate_fn(
        self, samples: List
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Assemble trajectories into batches with 0-padding.
        Args:
            samples: list of sampled trajectories (list of outputs of __getitem__)
        Returns:
            (starred dimensions have different values from one batch to the next but the ones with the same name are consistent within the batch)
            batch : ((batch_size, num_agents*, num_steps, state_dim),           # past trajectories of all agents in the scene
                     (batch_size, num_agents*, num_steps),                      # mask past False where past trajectories are padding data
                     (batch_size, num_agents*, num_steps_future, state_dim),    # future trajectory
                     (batch_size, num_agents*, num_steps_future),               # mask future False where future trajectories are padding data
                     (batch_size, num_agents*, num_steps_future),               # mask loss False where future trajectories are not to be predicted
                     (batch_size, num_objects*, object_seq_len*, map_state_dim),# map object sequences in the scene
                     (batch_size, num_objects*, object_seq_len*),               # mask map False where map objects are padding data
                     (batch_size, num_agents*, state_dim),                      # position offset of all agents relative to ego at present time
                     (batch_size, num_steps, state_dim),                        # ego past trajectory
                     (batch_size, num_steps_future, state_dim))                 # ego future trajectory
        """
        max_n_vehicle = 50
        max_n_lanes = 0
        for (
            coor,
            mask,
            mask_loss,
            lanes,
            mask_lanes,
            lane_states,
            mean_angle,
            mean_pos,
            idx,
        ) in samples:
            # time_len_coor = self._count_last_obs(coor, hist_len)
            # num_vehicle = np.sum(time_len_coor > self.min_num_obs)
            num_vehicle = coor.shape[1]
            num_lanes = lanes.shape[1]
            max_n_vehicle = max(num_vehicle, max_n_vehicle)
            max_n_lanes = max(num_lanes, max_n_lanes)
        if max_n_vehicle <= 0:
            raise RuntimeError
        data_batch = np.zeros(
            [self.time_len, len(samples), max_n_vehicle, self.state_dim]
        )
        mask_batch = np.zeros([self.time_len, len(samples), max_n_vehicle])
        mask_loss_batch = np.zeros([self.time_len, len(samples), max_n_vehicle])
        lane_batch = np.zeros(
            [self.max_size_lane, len(samples), max_n_lanes, self.map_state_dim]
        )
        mask_lane_batch = np.zeros([self.max_size_lane, len(samples), max_n_lanes])
        mean_angle_batch = np.zeros([len(samples)])
        mean_pos_batch = np.zeros([len(samples), 2])
        tag_list = np.zeros([len(samples)])
        idx_list = [0 for _ in range(len(samples))]

        for sample_ind, (
            coor,
            mask,
            mask_loss,
            lanes,
            mask_lanes,
            lane_states,
            mean_angle,
            mean_pos,
            idx,
        ) in enumerate(samples):
            data_batch[:, sample_ind, : coor.shape[1], :] = coor[: self.time_len, :, :]
            mask_batch[:, sample_ind, : mask.shape[1]] = mask[: self.time_len, :]
            mask_loss_batch[:, sample_ind, : mask.shape[1]] = mask_loss[
                : self.time_len, :
            ]
            lane_batch[: lanes.shape[0], sample_ind, : lanes.shape[1], :2] = lanes
            if lane_states is not None:
                lane_states = repeat(
                    lane_states[:, : self.hist_len],
                    "objects time features -> one objects (time features)",
                    one=1,
                )
                lane_batch[
                    : lanes.shape[0], sample_ind, : lanes.shape[1], 2:
                ] = lane_states
            mask_lane_batch[
                : mask_lanes.shape[0], sample_ind, : mask_lanes.shape[1]
            ] = mask_lanes
            mean_angle_batch[sample_ind] = mean_angle
            mean_pos_batch[sample_ind, :] = mean_pos
            # tag_list[sample_ind] = self.dataset[idx]["tag"]
            idx_list[sample_ind] = idx

        data_batch = torch.from_numpy(data_batch.astype("float32"))
        mask_batch = torch.from_numpy(mask_batch.astype("bool"))
        lane_batch = torch.from_numpy(lane_batch.astype("float32"))
        mask_lane_batch = torch.from_numpy(mask_lane_batch.astype("bool"))
        mean_pos_batch = torch.from_numpy(mean_pos_batch.astype("float32"))
        mask_loss_batch = torch.from_numpy(mask_loss_batch.astype("bool"))

        data_batch = rearrange(
            data_batch, "time batch agents features -> batch agents time features"
        )
        mask_batch = rearrange(mask_batch, "time batch agents -> batch agents time")
        mask_loss_batch = rearrange(
            mask_loss_batch, "time batch agents -> batch agents time"
        )
        lane_batch = rearrange(
            lane_batch,
            "object_seq_len batch objects features-> batch objects object_seq_len features",
        )
        mask_lane_batch = rearrange(
            mask_lane_batch,
            "object_seq_len batch objects -> batch objects object_seq_len",
        )

        # The two first agents are the ones interacting, others are sorted by distance from the first agent
        # Objects are also sorted by distance from the first agent
        # Therefore, the limits in number, max_num_agents and max_num_objects can be seen as adaptative distance limits.

        if not self.input_angle:
            data_batch = torch.cat((data_batch[..., :2], data_batch[..., 3:]), dim=-1)
        traj_past = data_batch[:, : self.max_num_agents, : self.hist_len, :]
        mask_past = mask_batch[:, : self.max_num_agents, : self.hist_len]
        traj_fut = data_batch[
            :, : self.max_num_agents, self.hist_len : self.hist_len + self.fut_len, :
        ]
        mask_fut = mask_batch[
            :, : self.max_num_agents, self.hist_len : self.hist_len + self.fut_len
        ]
        ego_past = data_batch[:, 0:1, : self.hist_len, :]
        ego_fut = data_batch[:, 0:1, self.hist_len : self.hist_len + self.fut_len, :]

        lane_batch = lane_batch[:, : self.max_num_objects]
        mask_lane_batch = mask_lane_batch[:, : self.max_num_objects]

        # Define what to predict (could be from Waymo's label of what to predict or the other agent that interacts with the ego...)
        mask_loss_batch = torch.logical_and(
            mask_loss_batch[
                :, : self.max_num_agents, self.hist_len : self.hist_len + self.fut_len
            ],
            mask_past.any(-1, keepdim=True),
        )
        # Remove all other agents so the model should only predict the first one
        mask_loss_batch[:, 0] = False
        mask_loss_batch[:, 2:] = False

        # Normalize...
        # traj_past = self.fill_past(traj_past, mask_past)
        dynamic_state_size = 5 if self.input_angle else 4
        offset_batch = traj_past[..., -1, :dynamic_state_size].clone()
        traj_past[..., :dynamic_state_size] = traj_past[
            ..., :dynamic_state_size
        ] - offset_batch.unsqueeze(-2)
        traj_fut[..., :dynamic_state_size] = traj_fut[
            ..., :dynamic_state_size
        ] - offset_batch.unsqueeze(-2)

        return (
            traj_past,
            mask_past,
            traj_fut,
            mask_fut,
            mask_loss_batch,
            lane_batch,
            mask_lane_batch,
            offset_batch,
            ego_past,
            ego_fut,
        )


class WaymoDataloaders:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def sample_dataloader(self) -> DataLoader:
        """Setup and return sample DataLoader

        Returns:
            DataLoader: sample DataLoader
        """
        dataset = WaymoDataset(self.cfg, "sample")
        sample_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=True,
        )
        return sample_loader

    def val_dataloader(
        self, drop_last=True, shuffle=False, input_angle=True
    ) -> DataLoader:
        """Setup and return validation DataLoader

        Returns:
            DataLoader: validation DataLoader
        """
        dataset = WaymoDataset(self.cfg, "val", input_angle)
        val_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )
        torch.cuda.empty_cache()
        return val_loader

    def train_dataloader(
        self, drop_last=True, shuffle=True, input_angle=True
    ) -> DataLoader:
        """Setup and return training DataLoader

        Returns:
            DataLoader: training DataLoader
        """
        dataset = WaymoDataset(self.cfg, "train", input_angle)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=dataset.collate_fn,
            drop_last=drop_last,
        )
        torch.cuda.empty_cache()
        return train_loader

    def test_dataloader(self) -> DataLoader:
        """Setup and return test DataLoader

        Returns:
            DataLoader: test DataLoader
        """
        raise NotImplementedError("The waymo dataloader cannot load test samples yet.")

    @staticmethod
    def unnormalize_trajectory(
        input: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalize trajectory by adding offset to input

        Args:
            input : (..., (n_sample), num_steps_future, state_dim) tensor of future
            trajectory y
            offset : (..., state_dim) tensor of offset to add to y

        Returns:
            Unnormalized trajectory that has the same size as input
        """
        assert offset.ndim == 3
        batch_size, num_agents = offset.shape[:2]
        offset_state_dim = offset.shape[-1]
        assert offset_state_dim <= input.shape[-1]
        assert input.shape[0] == batch_size
        assert input.shape[1] == num_agents
        input_copy = input.clone()

        input_copy[..., :offset_state_dim] = input_copy[
            ..., :offset_state_dim
        ] + offset[..., : input.shape[-1]].reshape(
            [batch_size, num_agents, *([1] * (input.ndim - 3)), offset_state_dim]
        )
        return input_copy
