from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from mmcv import Config
import numpy as np
import torch
from torch import Tensor


def masked_min_torch(x, mask=None, dim=None):
    if mask is not None:
        x = torch.masked_fill(x, torch.logical_not(mask), float("inf"))
    if dim is None:
        return torch.min(x)
    else:
        return torch.min(x, dim=dim)[0]


def masked_max_torch(x, mask=None, dim=None):
    if mask is not None:
        x = torch.masked_fill(x, torch.logical_not(mask), float("-inf"))
    if dim is None:
        return torch.max(x)
    else:
        return torch.max(x, dim=dim)[0]


def get_masked_discounted_mean_torch(discount_factor=0.95):
    def masked_discounted_mean_torch(x, mask=None, dim=None):
        discount_tensor = torch.full(x.shape, discount_factor, device=x.device)
        discount_tensor = torch.cumprod(discount_tensor, dim=-2)
        if mask is not None:
            x = torch.masked_fill(x, torch.logical_not(mask), 0)
            if dim is None:
                assert mask.any()
                return (x * discount_tensor).sum() / (mask * discount_tensor).sum()
            else:
                return (x * discount_tensor).sum(dim) / (mask * discount_tensor).sum(
                    dim
                ).clamp_min(1)
        else:
            if dim is None:
                return (x * discount_tensor).sum() / discount_tensor.sum()
            else:
                return (x * discount_tensor).sum(dim) / discount_tensor.sum(dim)

    return masked_discounted_mean_torch


def masked_mean_torch(x, mask=None, dim=None):
    if mask is not None:
        x = torch.masked_fill(x, torch.logical_not(mask), 0)
        if dim is None:
            assert mask.any()
            return x.sum() / mask.sum()
        else:
            return x.sum(dim) / mask.sum(dim).clamp_min(1)
    else:
        if dim is None:
            return x.mean()
        else:
            return x.mean(dim)


def get_discounted_mean_np(discount_factor=0.95):
    def discounted_mean_np(x, axis=None):
        discount_tensor = np.full(x.shape, discount_factor)
        discount_tensor = np.cumprod(discount_tensor, axis=-2)
        if axis is None:
            return (x * discount_tensor).sum() / discount_tensor.sum()
        else:
            return (x * discount_tensor).sum(axis) / discount_tensor.sum(axis)

    return discounted_mean_np


def get_masked_reduce_np(reduce_function):
    def masked_reduce_np(x, mask=None, axis=None):
        if mask is not None:
            x = np.ma.array(x, mask=np.logical_not(mask))
            return reduce_function(x, axis=axis)
        else:
            return reduce_function(x, axis=axis)

    return masked_reduce_np


@dataclass
class CostParams:
    scale: float
    reduce: str
    discount_factor: float

    @staticmethod
    def from_config(cfg: Config):
        return CostParams(
            scale=cfg.cost_scale,
            reduce=cfg.cost_reduce,
            discount_factor=cfg.discount_factor,
        )


class BaseCostTorch:
    """Base cost class defining reduce strategy and basic parameters.
    Its __call__ definition is only a dummy example returning zeros, this class is intended to be
    inherited from and __call__ redefined with an actual cost between the inputs.
    """

    def __init__(self, params: CostParams) -> None:
        super().__init__()
        self._reduce_fun = params.reduce
        self.scale = params.scale

        reduce_fun_torch_dict = {
            "min": masked_min_torch,
            "max": masked_max_torch,
            "mean": masked_mean_torch,
            "discounted_mean": get_masked_discounted_mean_torch(params.discount_factor),
            "now": self.get_now,
            "final": self.get_final,
        }

        self._reduce_fun = reduce_fun_torch_dict[params.reduce]
        
    def get_now(self, x, *args, **kwargs):
        return x[..., 0]
    
    def get_final(self, x, *args, **kwargs):
        return x[..., -1]

    @property
    def distance_bandwidth(self):
        return 1

    @property
    def time_bandwidth(self):
        return 1

    def __call__(
        self,
        x1: Tensor,
        x2: Tensor,
        v1: Tensor,
        v2: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Any]:
        """Compute the cost from given positions x1, x2 and velocities v1, v2
           The base cost only returns 0 cost, use costs that inherit from this to compute an actual cost.

        Args:
            x1 (some shape, num_steps, 2): positions of the first agent
            x2 (some shape, num_steps, 2): positions of the second agent
            v1 (some shape, num_steps, 2): velocities of the first agent
            v2 (some shape, num_steps, 2): velocities of the second agent
            mask (some_shape, num_steps, 2): mask set to True where the cost should be computed

        Returns:
            (some_shape) cost for the compared states of agent 1 and agent 2, as well as any
            supplementary cost-related information
        """
        return (
            self._reduce_fun(torch.zeros_like(x2[..., 0]), mask, dim=-1),
            None,
        )


class BaseCostNumpy:
    """Base cost class defining reduce strategy and basic parameters.
    Its __call__ definition is only a dummy example returning zeros, this class is intended to be
    inherited from and __call__ redefined with an actual cost between the inputs.
    """

    def __init__(self, params: CostParams) -> None:
        super().__init__()
        self._reduce_fun = params.reduce
        self.scale = params.scale

        reduce_fun_np_dict = {
            "min": get_masked_reduce_np(np.min),
            "max": get_masked_reduce_np(np.max),
            "mean": get_masked_reduce_np(np.mean),
            "discounted_mean": get_masked_reduce_np(
                get_discounted_mean_np(params.discount_factor)
            ),
            "now": get_masked_reduce_np(self.get_now),
            "final": get_masked_reduce_np(self.get_final),
        }
        self._reduce_fun = reduce_fun_np_dict[params.reduce]
        
    def get_now(self, x, *args, **kwargs):
        return x[..., 0]
    
    def get_final(self, x, *args, **kwargs):
        return x[..., -1]

    @property
    def distance_bandwidth(self):
        return 1

    @property
    def time_bandwidth(self):
        return 1

    def __call__(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Compute the cost from given positions x1, x2 and velocities v1, v2
           The base cost only returns 0 cost, use costs that inherit from this to compute an actual cost.

        Args:
            x1 (some shape, num_steps, 2): positions of the first agent
            x2 (some shape, num_steps, 2): positions of the second agent
            v1 (some shape, num_steps, 2): velocities of the first agent
            v2 (some shape, num_steps, 2): velocities of the second agent
            mask (some_shape, num_steps, 2): mask set to True where the cost should be computed

        Returns:
            (some_shape) cost for the compared states of agent 1 and agent 2, as well as any
            supplementary cost-related information
        """
        return (
            self._reduce_fun(np.zeros_like(x2[..., 0]), mask, axis=-1),
            None,
        )


@dataclass
class DistanceCostParams(CostParams):
    bandwidth: float

    @staticmethod
    def from_config(cfg: Config):
        return DistanceCostParams(
            scale=cfg.cost_scale,
            reduce=cfg.cost_reduce,
            bandwidth=cfg.distance_bandwidth,
            discount_factor=cfg.discount_factor,
        )


class DistanceCostTorch(BaseCostTorch):
    def __init__(self, params: DistanceCostParams) -> None:
        super().__init__(params)
        self._bandwidth = params.bandwidth

    @property
    def distance_bandwidth(self):
        return self._bandwidth

    def __call__(
        self, x1: Tensor, x2: Tensor, *args, mask: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns a cost estimation based on distance. Also returns distances between ego and pedestrians.
        Args:
            x1: First agent trajectory
            x2: Second agent trajectory
            mask: True where cost should be computed
        Returns:
            cost, distance_to_collision
        """

        dist = torch.square(x2 - x1).sum(-1)
        if mask is not None:
            dist = torch.masked_fill(dist, torch.logical_not(mask), 1e9)
        cost = torch.exp(-dist / (2 * self._bandwidth))
        return self.scale * self._reduce_fun(cost, mask=mask, dim=-1), dist


class DistanceCostNumpy(BaseCostNumpy):
    def __init__(self, params: DistanceCostParams) -> None:
        super().__init__(params)
        self._bandwidth = params.bandwidth

    @property
    def distance_bandwidth(self):
        return self._bandwidth

    def __call__(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        *args,
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a cost estimation based on distance. Also returns distances between ego and pedestrians.
        Args:
            x1: First agent trajectory
            x2: Second agent trajectory
            mask: True where cost should be computed
        Returns:
            cost, distance_to_collision
        """
        dist = np.square(x2 - x1).sum(-1)
        if mask is not None:
            dist = np.where(mask, dist, 1e9)
        cost = np.exp(-dist / (2 * self._bandwidth))
        return self.scale * self._reduce_fun(cost, mask=mask, axis=-1), dist


@dataclass
class TTCCostParams(CostParams):
    distance_bandwidth: float
    time_bandwidth: float
    min_velocity_diff: float

    @staticmethod
    def from_config(cfg: Config):
        return TTCCostParams(
            scale=cfg.cost_scale,
            reduce=cfg.cost_reduce,
            distance_bandwidth=cfg.distance_bandwidth,
            time_bandwidth=cfg.time_bandwidth,
            min_velocity_diff=cfg.min_velocity_diff,
            discount_factor=cfg.discount_factor,
        )


class TTCCostTorch(BaseCostTorch):
    def __init__(self, params: TTCCostParams) -> None:
        super().__init__(params)
        self._d_bw = params.distance_bandwidth
        self._t_bw = params.time_bandwidth
        self._min_v = params.min_velocity_diff

    @property
    def distance_bandwidth(self):
        return self._d_bw

    @property
    def time_bandwidth(self):
        return self._t_bw

    def __call__(
        self,
        x1: Tensor,
        x2: Tensor,
        v1: Tensor,
        v2: Tensor,
        *args,
        mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Returns a cost estimation based on time to collision and distance to collision.
        Also returns the estimated time to collision, and the imaginary part of the time to collision.
        Args:
            x1: (some_shape, sequence_length, feature_shape) Initial position of the first agent
            x2: (some_shape, sequence_length, feature_shape) Initial position of the second agent
            v1: (some_shape, sequence_length, feature_shape) Velocity of the first agent
            v2: (some_shape, sequence_length, feature_shape) Velocity of the second agent
            mask: (some_shape, sequence_length) True where cost should be computed
        Returns:
            cost, (time_to_collision, distance_to_collision)
        """
        pos_diff = x1 - x2
        velocity_diff = v1 - v2

        dx = pos_diff[..., 0]
        dy = pos_diff[..., 1]
        vx = velocity_diff[..., 0]
        vy = velocity_diff[..., 1]

        speed_diff = (
            torch.square(velocity_diff).sum(-1).clamp(self._min_v * self._min_v, None)
        )

        TTC = -(dx * vx + dy * vy) / speed_diff

        distance_TTC = torch.where(
            TTC < 0,
            torch.sqrt(dx * dx + dy * dy),
            torch.abs(vy * dx - vx * dy) / torch.sqrt(speed_diff),
        )
        TTC = torch.relu(TTC)
        if mask is not None:
            TTC = torch.masked_fill(TTC, torch.logical_not(mask), 1e9)
            distance_TTC = torch.masked_fill(distance_TTC, torch.logical_not(mask), 1e9)

        cost = self.scale * self._reduce_fun(
            torch.exp(
                -torch.square(TTC) / (2 * self._t_bw)
                - torch.square(distance_TTC) / (2 * self._d_bw)
            ),
            mask=mask,
            dim=-1,
        )

        return cost, (TTC, distance_TTC)


class TTCCostNumpy(BaseCostNumpy):
    def __init__(self, params: TTCCostParams) -> None:
        super().__init__(params)
        self._d_bw = params.distance_bandwidth
        self._t_bw = params.time_bandwidth
        self._min_v = params.min_velocity_diff

    @property
    def distance_bandwidth(self):
        return self._d_bw

    @property
    def time_bandwidth(self):
        return self._t_bw

    def __call__(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        *args,
        mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a cost estimation based on time to collision and distance to collision.
        Also returns the estimated time to collision, and the imaginary part of the time to collision.
        Args:
            x1: (some_shape, sequence_length, feature_shape) Initial position of the first agent
            x2: (some_shape, sequence_length, feature_shape) Initial position of the second agent
            v1: (some_shape, sequence_length, feature_shape) Velocity of the first agent
            v2: (some_shape, sequence_length, feature_shape) Velocity of the second agent
            mask: (some_shape, sequence_length) True where cost should be computed
        Returns:
            cost, (time_to_collision, distance_to_collision)
        """
        pos_diff = x1 - x2
        velocity_diff = v1 - v2

        dx = pos_diff[..., 0]
        dy = pos_diff[..., 1]
        vx = velocity_diff[..., 0]
        vy = velocity_diff[..., 1]

        speed_diff = np.maximum(
            np.square(velocity_diff).sum(-1), self._min_v * self._min_v
        )

        TTC = -(dx * vx + dy * vy) / speed_diff
        distance_TTC = np.where(
            TTC < 0,
            np.sqrt(dx * dx + dy * dy),
            np.abs(vy * dx - vx * dy) / np.sqrt(speed_diff),
        )
        TTC = np.where(
            TTC < 0,
            0,
            TTC,
        )
        if mask is not None:
            TTC = np.where(mask, TTC, 1e9)
            distance_TTC = np.where(mask, TTC, 1e9)

        cost = self.scale * self._reduce_fun(
            np.exp(
                -np.square(TTC) / (2 * self._t_bw)
                - np.square(distance_TTC) / (2 * self._d_bw)
            ),
            mask=mask,
            axis=-1,
        )
        return cost, (TTC, distance_TTC)


def compute_v_from_x(x: Tensor, y: Tensor, dt: float):
    """
    Computes the velocity from the position and the time difference.
    Args:
        x: (some_shape, past_time_sequence, features) Past positions of the agents
        y: (some_shape, future_time_sequence, features) Future positions of the agents
        dt: Time difference
    Returns:
        v: (some_shape, future_time_sequence, features) Velocity of the agents
    """
    v = (y[..., 1:, :2] - y[..., :-1, :2]) / dt
    v_0 = (y[..., 0:1, :2] - x[..., -1:, :2]) / dt
    v = torch.cat((v_0, v), -2)
    return v


def get_cost(
    cost_function: BaseCostTorch,
    x: torch.Tensor,
    y_samples: torch.Tensor,
    offset: torch.Tensor,
    x_ego: torch.Tensor,
    y_ego: torch.Tensor,
    dt: float,
    unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cost samples from predicted future trajectories

    Args:
        cost_function: Cost function to use
        x: (batch_size, n_agents, num_steps, state_dim) normalized tensor of history
        y_samples: (batch_size, n_agents, n_samples, num_steps_future, state_dim) normalized tensor of predicted
            future trajectory samples
        offset: (batch_size, n_agents, state_dim) offset position from ego
        x_ego: (batch_size, 1, num_steps, state_dim) tensor of ego history
        y_ego: (batch_size, 1, num_steps_future, state_dim) tensor of ego future trajectory
        dt: time step in trajectories
        unnormalizer: function that takes in a trajectory and an offset and that outputs the
                      unnormalized trajectory
        mask: tensor indicating where to compute the cost
    Returns:
        torch.Tensor: (batch_size, n_agents, n_samples) cost tensor
    """
    x = unnormalizer(x, offset)
    y_samples = unnormalizer(y_samples, offset)
    if offset.shape[1] > 1:
        x_ego = unnormalizer(x_ego, offset[:, 0:1])
        y_ego = unnormalizer(y_ego, offset[:, 0:1])

    min_dim = min(x.shape[-1], y_samples.shape[-1])
    x = x[..., :min_dim]
    y_samples = y_samples[..., :min_dim]
    x_ego = x_ego[..., :min_dim]
    y_ego = y_ego[..., :min_dim]
    assert x_ego.ndim == y_ego.ndim
    if y_samples.shape[-1] < 5:
        v_samples = compute_v_from_x(x.unsqueeze(-3), y_samples, dt)
    else:
        v_samples = y_samples[..., 3:5]

    if y_ego.shape[-1] < 5:
        v_ego = compute_v_from_x(x_ego, y_ego, dt)
    else:
        v_ego = y_ego[..., 3:5]
    if mask is not None:
        mask = torch.cat(
            (mask[..., 0:1], torch.logical_and(mask[..., 1:], mask[..., :-1])), -1
        )

    cost, _ = cost_function(
        x1=y_ego.unsqueeze(-3),
        x2=y_samples,
        v1=v_ego.unsqueeze(-3),
        v2=v_samples,
        mask=mask,
    )
    return cost
