import numpy as np
import torch
from torch import Tensor
from typing import Union


class RandomPedestrians:
    """
    Batched random pedestrians.
    There are two types of pedestrians, slow and fast ones.
    Each pedestrian type is walking mainly at its constant favored speed but at each time step there is a probability that it changes its pace.

    Args:
        batch_size: int number of scenes in the batch
        dt: float time step to use in the trajectory sequence
        fast_speed: float fast walking speed for the random pedestrian in meters/seconds
        slow_speed: float slow walking speed for the random pedestrian in meters/seconds
        p_change_pace: float probability that a slow (resp. fast) pedestrian walk at fast_speed (resp. slow_speed) at each time step
        proportion_fast: float proportion of the pedestrians that are mainly walking at fast_speed
        is_torch: bool set to True to produce Tensor batches and to False to produce numpy arrays
    """

    def __init__(
        self,
        batch_size: int,
        dt: float = 0.1,
        fast_speed: float = 2,
        slow_speed: float = 1,
        p_change_pace: float = 0.1,
        proportion_fast: float = 0.5,
        is_torch: bool = False,
    ) -> None:

        self.is_torch = is_torch
        self.fast_speed: float = fast_speed
        self.slow_speed: float = slow_speed
        self.dt: float = dt
        self.p_change_pace: float = p_change_pace
        self.batch_size: int = batch_size

        self.propotion_fast: float = proportion_fast
        if self.is_torch:
            self.is_fast_type: Tensor = torch.from_numpy(
                np.random.binomial(1, self.propotion_fast, [batch_size, 1, 1]).astype(
                    "float32"
                )
            )
            self.is_currently_fast: Tensor = self.is_fast_type.clone()
            self.initial_position: Tensor = torch.zeros([batch_size, 1, 2])
            self.position: Tensor = self.initial_position.clone()
            self._angle: Tensor = (2 * torch.rand(batch_size, 1) - 1) * np.pi
            self.unit_velocity: Tensor = torch.stack(
                (torch.cos(self._angle), torch.sin(self._angle)), -1
            )
        else:
            self.is_fast_type: np.ndarray = np.random.binomial(
                1, self.propotion_fast, [batch_size, 1, 1]
            )
            self.is_currently_fast: np.ndarray = self.is_fast_type.copy()
            self.initial_position: np.ndarray = np.zeros([batch_size, 1, 2])
            self.position: np.ndarray = self.initial_position.copy()
            self._angle: np.ndarray = np.random.uniform(-np.pi, np.pi, (batch_size, 1))
            self.unit_velocity: np.ndarray = np.stack(
                (np.cos(self._angle), np.sin(self._angle)), -1
            )

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle: Union[np.ndarray, torch.Tensor]):
        assert self.batch_size == angle.shape[0]
        if self.is_torch:
            assert isinstance(angle, torch.Tensor)
            self._angle = angle
            self.unit_velocity = torch.stack(
                (torch.cos(self._angle), torch.sin(self._angle)), -1
            )
        else:
            assert isinstance(angle, np.ndarray)
            self._angle = angle
            self.unit_velocity = np.stack(
                (np.cos(self._angle), np.sin(self._angle)), -1
            )

    def step(self) -> None:
        """
        Forward one time step, update the speed selection and the current position.
        """
        self.update_speed()
        self.update_position()

    def update_speed(self) -> None:
        """
        Update the speed as a random selection between favored speed and the other speed with probability self.p_change_pace.
        """
        if self.is_torch:
            do_flip = (
                torch.from_numpy(
                    np.random.binomial(1, self.p_change_pace, self.batch_size).astype(
                        "float32"
                    )
                )
                == 1
            )
            self.is_currently_fast = self.is_fast_type.clone()
        else:
            do_flip = np.random.binomial(1, self.p_change_pace, self.batch_size) == 1
            self.is_currently_fast = self.is_fast_type.copy()
        self.is_currently_fast[do_flip] = 1 - self.is_fast_type[do_flip]

    def update_position(self) -> None:
        """
        Update the position as current position + time_step*speed*(cos(angle), sin(angle))
        """
        self.position += (
            self.dt
            * (
                self.slow_speed
                + (self.fast_speed - self.slow_speed) * self.is_currently_fast
            )
            * self.unit_velocity
        )

    def travel_distance(self) -> Union[np.ndarray, Tensor]:
        """
        Return the travel distance between initial position and current position.
        """
        if self.is_torch:
            return torch.sqrt(
                torch.sum(torch.square(self.position - self.initial_position), -1)
            )
        else:
            return np.sqrt(np.sum(np.square(self.position - self.initial_position), -1))

    def get_final_position(self, time: float) -> Union[np.ndarray, Tensor]:
        """
        Return a sample of pedestrian final positions using their speed distribution.
        (This is stochastic, different samples will produce different results).
        Args:
            time: The final time at which to get the position
        Returns:
            The batch of final positions
        """
        num_steps = int(round(time / self.dt))
        if self.is_torch:
            cumulative_change_state = torch.from_numpy(
                np.random.binomial(
                    num_steps, self.p_change_pace, [self.batch_size, 1, 1]
                ).astype("float32")
            )
        else:
            cumulative_change_state = np.random.binomial(
                num_steps, self.p_change_pace, [self.batch_size, 1, 1]
            )

        num_fast_steps = (
            num_steps - 2 * cumulative_change_state
        ) * self.is_fast_type + cumulative_change_state

        return self.position + self.unit_velocity * self.dt * (
            self.slow_speed * num_steps
            + (self.fast_speed - self.slow_speed) * num_fast_steps
        )
