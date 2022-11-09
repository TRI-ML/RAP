from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class SceneDataLoaders:
    """
    This class loads a scene dataset and pre-process it (normalization, unnormalization)

    Args:
        state_dim : dimension of the observed state (2 for x,y position observation)
        num_steps : number of observed steps
        num_steps_future : number of steps in the future
        batch_size: set data loader with this batch size
        data_train: training dataset
        data_val: validation dataset
        data_test: test dataset
        num_workers: number of workers to use for data loading
    """

    def __init__(
        self,
        state_dim: int,
        num_steps: int,
        num_steps_future: int,
        batch_size: int,
        data_train: torch.Tensor,
        data_val: torch.Tensor,
        data_test: torch.Tensor,
        num_workers: int = 0,
    ):
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._state_dim = state_dim
        self._num_steps = num_steps
        self._num_steps_future = num_steps_future

        self._setup_datasets(data_train, data_val, data_test)

    def train_dataloader(self, shuffle=True, drop_last=True) -> DataLoader:
        """Setup and return training DataLoader

        Returns:
            DataLoader: training DataLoader
        """
        data_size = self._data_train_past.shape[0]
        # This is a didactic data loader that only defines minimalistic inputs.
        # This dataloader adds some empty tensors and ones to match the expected format with masks and map information.
        train_loader = DataLoader(
            dataset=TensorDataset(
                self._data_train_past,
                torch.ones_like(self._data_train_past[..., 0]),  # Mask past
                self._data_train_fut,
                torch.ones_like(self._data_train_fut[..., 0]),  # Mask fut
                torch.ones_like(self._data_train_fut[..., 0]),  # Mask loss
                torch.empty(
                    data_size, 1, 0, 0, device=self._data_train_past.device
                ),  # Map
                torch.empty(
                    data_size, 1, 0, device=self._data_train_past.device
                ),  # Mask map
                self._offset_train,
                self._data_train_ego_past,
                self._data_train_ego_fut,
            ),
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers,
        )
        return train_loader

    def val_dataloader(self, shuffle=False, drop_last=False) -> DataLoader:
        """Setup and return validation DataLoader

        Returns:
            DataLoader: validation DataLoader
        """
        data_size = self._data_val_past.shape[0]
        # This is a didactic data loader that only defines minimalistic inputs.
        # This dataloader adds some empty tensors and ones to match the expected format with masks and map information.
        val_loader = DataLoader(
            dataset=TensorDataset(
                self._data_val_past,
                torch.ones_like(self._data_val_past[..., 0]),  # Mask past
                self._data_val_fut,
                torch.ones_like(self._data_val_fut[..., 0]),  # Mask fut
                torch.ones_like(self._data_val_fut[..., 0]),  # Mask loss
                torch.zeros(
                    data_size, 1, 0, 0, device=self._data_val_past.device
                ),  # Map
                torch.ones(
                    data_size, 1, 0, device=self._data_val_past.device
                ),  # Mask map
                self._offset_val,
                self._data_val_ego_past,
                self._data_val_ego_fut,
            ),
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers,
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        """Setup and return test DataLoader

        Returns:
            DataLoader: test DataLoader
        """
        data_size = self._data_test_past.shape[0]
        # This is a didactic data loader that only defines minimalistic inputs.
        # This dataloader adds some empty tensors and ones to match the expected format with masks and map information.
        test_loader = DataLoader(
            dataset=TensorDataset(
                self._data_test_past,
                torch.ones_like(self._data_test_past[..., 0]),  # Mask
                torch.zeros(
                    data_size, 0, 1, 0, device=self._data_test_past.device
                ),  # Map
                torch.ones(
                    data_size, 0, 1, device=self._data_test_past.device
                ),  # Mask map
                self._offset_test,
                self._data_test_ego_past,
                self._data_test_ego_fut,
            ),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
        return test_loader

    def _setup_datasets(
        self, data_train: torch.Tensor, data_val: torch.Tensor, data_test: torch.Tensor
    ):
        """Setup datasets: normalize and split into past future
        Args:
            data_train: training dataset
            data_val: validation dataset
            data_test: test dataset
        """
        data_train, data_train_ego = data_train[0], data_train[1]
        data_val, data_val_ego = data_val[0], data_val[1]
        data_test, data_test_ego = data_test[0], data_test[1]

        data_train, self._offset_train = self.normalize_trajectory(data_train)
        data_val, self._offset_val = self.normalize_trajectory(data_val)
        data_test, self._offset_test = self.normalize_trajectory(data_test)
        # This is a didactic data loader that only defines minimalistic inputs.
        # An extra dimension is added to account for the number of agents in the scene.
        # In this minimal input there is only one but the model using the data expects any number of agents.
        self._data_train_past, self._data_train_fut = self.split_trajectory(data_train)
        self._data_val_past, self._data_val_fut = self.split_trajectory(data_val)
        self._data_test_past, self._data_test_fut = self.split_trajectory(data_test)

        self._data_train_ego_past, self._data_train_ego_fut = self.split_trajectory(
            data_train_ego
        )
        self._data_val_ego_past, self._data_val_ego_fut = self.split_trajectory(
            data_val_ego
        )
        self._data_test_ego_past, self._data_test_ego_fut = self.split_trajectory(
            data_test_ego
        )

    def split_trajectory(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input trajectory into history and future

        Args:
            input : (batch_size, (n_agents), num_steps + num_steps_future, state_dim) tensor of
            entire trajectory [x, y]

        Returns:
            Tuple of history and future trajectories
        """
        assert (
            input.shape[-2] == self._num_steps + self._num_steps_future
        ), "trajectory length ({}) does not match the expected length".format(
            input.shape[-2]
        )
        assert (
            input.shape[-1] == self._state_dim
        ), "state dimension ({}) does no match the expected dimension".format(
            input.shape[-1]
        )

        input_history, input_future = torch.split(
            input, [self._num_steps, self._num_steps_future], dim=-2
        )
        return input_history, input_future

    @staticmethod
    def normalize_trajectory(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize input trajectory by subtracting initial state

        Args:
            input : (some_shape, n_agents, num_steps + num_steps_future, state_dim) tensor of
            entire trajectory [x, y], or (some_shape, num_steps, state_dim) tensor of history x

        Returns:
            Tuple of (normalized_trajectory, offset), where
            normalized_trajectory has the same dimension as the input and offset is a
            (some_shape, state_dim) tensor corresponding to the initial state
        """
        offset = input[..., 0, :].clone()

        return input - offset.unsqueeze(-2), offset

    @staticmethod
    def unnormalize_trajectory(
        input: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Unnormalize trajectory by adding offset to input

        Args:
            input : (some_shape, (n_sample), num_steps_future, state_dim) tensor of future
            trajectory y
            offset : (some_shape, 2 or 4 or 5) tensor of offset to add to y

        Returns:
            Unnormalized trajectory that has the same size as input
        """
        offset_dim = offset.shape[-1]
        assert input.shape[-1] >= offset_dim
        input_clone = input.clone()
        if offset.ndim == 2:
            batch_size, _ = offset.shape
            assert input_clone.shape[0] == batch_size

            input_clone[..., :offset_dim] = input_clone[
                ..., :offset_dim
            ] + offset.reshape(
                [batch_size, *([1] * (input_clone.ndim - 2)), offset_dim]
            )
        elif offset.ndim == 3:
            batch_size, num_agents, _ = offset.shape
            assert input_clone.shape[0] == batch_size
            assert input_clone.shape[1] == num_agents

            input_clone[..., :offset_dim] = input_clone[
                ..., :offset_dim
            ] + offset.reshape(
                [batch_size, num_agents, *([1] * (input_clone.ndim - 3)), offset_dim]
            )

        return input_clone
