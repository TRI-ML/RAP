from dataclasses import dataclass
from typing import Optional

import torch
from mmcv import Config


@dataclass
class TrackingCostParams:
    scale_longitudinal: float
    scale_lateral: float
    reduce: str

    @staticmethod
    def from_config(cfg: Config):
        return TrackingCostParams(
            scale_longitudinal=cfg.tracking_cost_scale_longitudinal,
            scale_lateral=cfg.tracking_cost_scale_lateral,
            reduce=cfg.tracking_cost_reduce,
        )


class TrackingCost:
    """Quadratic Trajectory Tracking Cost

    Args:
        params: tracking cost parameters
    """

    def __init__(self, params: TrackingCostParams) -> None:
        self.scale_longitudinal = params.scale_longitudinal
        self.scale_lateral = params.scale_lateral
        assert params.reduce in [
            "min",
            "max",
            "mean",
            "now",
            "final",
        ], "unsupported reduce type"
        self._reduce_fun_name = params.reduce

    def __call__(
        self,
        ego_position_trajectory: torch.Tensor,
        target_position_trajectory: torch.Tensor,
        target_velocity_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Computes quadratic tracking cost

        Args:
            ego_position_trajectory: (some_shape, num_some_steps, 2) tensor of ego
              position trajectory
            target_position_trajectory: (some_shape, num_some_steps, 2) tensor of
              ego target position trajectory
            target_velocity_trajectory: (some_shape, num_some_steps, 2) tensor of
              ego target velocity trajectory

        Returns:
            (some_shape) cost
        """
        cost_matrix = self._get_quadratic_cost_matrix(target_velocity_trajectory)
        cost = (
            (
                (ego_position_trajectory - target_position_trajectory).unsqueeze(-2)
                @ cost_matrix
                @ (ego_position_trajectory - target_position_trajectory).unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return self._reduce(cost, dim=-1)

    def _reduce(self, cost: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """Reduces the cost tensor based on self._reduce_fun_name

        Args:
            cost: cost tensor of some shape where the last dimension represents time
            dim (optional): tensor dimension to be reduced. Defaults to None.

        Returns:
            reduced cost tensor
        """
        if self._reduce_fun_name == "min":
            return torch.min(cost, dim=dim)[0] if dim is not None else torch.min(cost)
        if self._reduce_fun_name == "max":
            return torch.max(cost, dim=dim)[0] if dim is not None else torch.max(cost)
        if self._reduce_fun_name == "mean":
            return torch.mean(cost, dim=dim) if dim is not None else torch.mean(cost)
        if self._reduce_fun_name == "now":
            return cost[..., 0]
        if self._reduce_fun_name == "final":
            return cost[..., -1]

    def _get_quadratic_cost_matrix(
        self, target_velocity_trajectory: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Gets quadratic cost matrix based on target velocity direction per time step.
        If target velocity is 0 in norm, then all zero tensor is returned for that time step.

        Args:
            target_velocity_trajectory: (some_shape, num_some_steps, 2) tensor of
              ego target velocity trajectory
            eps (optional): small positive number to ensure numerical stability. Defaults to
              1e-8.

        Returns:
            (some_shape, num_some_steps, 2, 2) quadratic cost matrix
        """
        longitudinal_direction = (
            target_velocity_trajectory
            / (
                torch.linalg.norm(target_velocity_trajectory, dim=-1).unsqueeze(-1)
                + eps
            )
        ).unsqueeze(-1)
        rotation_90_deg = torch.Tensor([[[0.0, -1.0], [1.0, 0]]])
        lateral_direction = rotation_90_deg @ longitudinal_direction
        orthogonal_matrix = torch.cat(
            (longitudinal_direction, lateral_direction), dim=-1
        )
        eigen_matrix = torch.Tensor(
            [[[self.scale_longitudinal, 0.0], [0.0, self.scale_lateral]]]
        )
        cost_matrix = (
            orthogonal_matrix @ eigen_matrix @ orthogonal_matrix.transpose(-1, -2)
        )
        return cost_matrix
