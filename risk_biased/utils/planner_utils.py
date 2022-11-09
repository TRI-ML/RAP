from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from numpy import isin

import torch

from risk_biased.mpc_planner.planner_cost import TrackingCost
from risk_biased.utils.cost import BaseCostTorch
from risk_biased.utils.risk import AbstractMonteCarloRiskEstimator


def get_rotation_matrix(angle, device):
    c = torch.cos(angle)
    s = torch.sin(angle)
    rot_matrix = torch.stack(
        (torch.stack((c, s), -1), torch.stack((-s, c), -1)), -1
    ).to(device)
    return rot_matrix


class AbstractState(ABC):
    """
    State representation using an underlying tensor. Position, Velocity, and Angle can be accessed.
    """

    @property
    @abstractmethod
    def position(self) -> torch.Tensor:
        """Extract position information from the state tensor

        Returns:
            position_tensor of size (..., 2)
        """

    @property
    @abstractmethod
    def velocity(self) -> torch.Tensor:
        """Extract velocity information from the state tensor

        Returns:
            velocity_tensor of size (..., 2)
        """

    @property
    @abstractmethod
    def angle(self) -> torch.Tensor:
        """Extract velocity information from the state tensor

        Returns:
            velocity_tensor of size (..., 1)
        """

    @abstractmethod
    def get_states(self, dim: int) -> torch.Tensor:
        """Return the underlying states tensor with dim 2, 4 or 5 ([x, y], [x, y, vx, vy], or [x, y, angle, vx, vy])."""

    @abstractmethod
    def rotate(self, angle: float, in_place: bool) -> AbstractState:
        """Rotate the state by the given angle
        Args:
            angle: in radiants
            in_place: wether to change the object itself or return a rotated copy
        Returns:
            rotated self or rotated copy of self
        """

    @abstractmethod
    def translate(self, translation: torch.Tensor, in_place: bool) -> AbstractState:
        """Translate the state by the given tranlation
        Args:
            translation: translation vector in 2 dimensions
            in_place: wether to change the object itself or return a rotated copy
        """

    # Define overloading operators to behave as a tensor for some operations
    def __getitem__(self, key) -> AbstractState:
        """
        Use get item on the underlying tensor to get the item at the given key.
        Allways returns a velocity state so that if the underlying time sequence is reduced to one step, the velocity is still accessible.
        """
        if isinstance(key, int):
            key = (key, Ellipsis, slice(None, None, None))
        elif Ellipsis not in key:
            key = (*key, Ellipsis, slice(None, None, None))
        else:
            key = (*key, slice(None, None, None))

        return to_state(
            torch.cat(
                (
                    self.position[key],
                    self.velocity[key],
                ),
                dim=-1,
            ),
            self.dt,
        )

    @property
    def shape(self):
        return self._states.shape[:-1]


def to_state(in_tensor: torch.Tensor, dt: float) -> AbstractState:
    if in_tensor.shape[-1] == 2:
        return PositionSequenceState(in_tensor, dt)
    elif in_tensor.shape[-1] == 4:
        return PositionVelocityState(in_tensor, dt)
    else:
        assert in_tensor.shape[-1] > 4
        return PositionAngleVelocityState(in_tensor, dt)


class PositionSequenceState(AbstractState):
    """
    State representation with an underlying tensor defining only positions.
    """

    def __init__(self, states: torch.Tensor, dt: float) -> None:
        super().__init__()
        assert (
            states.shape[-1] == 2
        )  # Check that the input tensor defines only the position
        assert (
            states.ndim > 1 and states.shape[-2] > 1
        )  # Check that the input tensor defines a sequence of positions (otherwise velocity cannot be computed)
        self.dt = dt
        self._states = states.clone()

    @property
    def position(self) -> torch.Tensor:
        return self._states

    @property
    def velocity(self) -> torch.Tensor:
        vel = (self._states[..., 1:, :] - self._states[..., :-1, :]) / self.dt
        vel = torch.cat((vel[..., 0:1, :], vel), dim=-2)
        return vel.clone()

    @property
    def angle(self) -> torch.Tensor:
        vel = self.velocity
        angle = torch.arctan2(vel[..., 1:2], vel[..., 0:1])
        return angle

    def get_states(self, dim: int = 2) -> torch.Tensor:
        if dim == 2:
            return self._states.clone()
        elif dim == 4:
            return torch.cat((self._states.clone(), self.velocity), dim=-1)
        elif dim == 5:
            return torch.cat((self._states.clone(), self.angle, self.velocity), dim=-1)
        else:
            raise RuntimeError(f"State dimension must be either 2, 4, or 5. Got {dim}")

    def rotate(self, angle: float, in_place: bool = False) -> PositionSequenceState:
        """Rotate the state by the given angle in radiants"""
        rot_matrix = get_rotation_matrix(angle, self._states.device)
        if in_place:
            self._states = (rot_matrix @ self._states.unsqueeze(-1)).squeeze(-1)
            return self
        else:
            return to_state(
                (rot_matrix @ self._states.unsqueeze(-1).clone()).squeeze(-1), self.dt
            )

    def translate(
        self, translation: torch.Tensor, in_place: bool = False
    ) -> PositionSequenceState:
        """Translate the state by the given tranlation"""
        if in_place:
            self._states[..., :2] += translation.expand_as(self._states[..., :2])
            return self
        else:
            return to_state(
                self._states[..., :2].clone()
                + translation.expand_as(self._states[..., :2]),
                self.dt,
            )


class PositionVelocityState(AbstractState):
    """
    State representation with an underlying tensor defining position and velocity.
    """

    def __init__(self, states: torch.Tensor, dt) -> None:
        super().__init__()
        assert states.shape[-1] == 4
        self._states = states.clone()
        self.dt = dt

    @property
    def position(self) -> torch.Tensor:
        return self._states[..., :2]

    @property
    def velocity(self) -> torch.Tensor:
        return self._states[..., 2:4]

    @property
    def angle(self) -> torch.Tensor:
        vel = self.velocity
        angle = torch.arctan2(vel[..., 1:2], vel[..., 0:1])
        return angle

    def get_states(self, dim: int = 4) -> torch.Tensor:
        if dim == 2:
            return self._states[..., :2].clone()
        elif dim == 4:
            return self._states.clone()
        elif dim == 5:
            return torch.cat(
                (
                    self._states[..., :2].clone(),
                    self.angle,
                    self._states[..., 2:].clone(),
                ),
                dim=-1,
            )
        else:
            raise RuntimeError(f"State dimension must be either 2, 4, or 5. Got {dim}")

    def rotate(
        self, angle: torch.Tensor, in_place: bool = False
    ) -> PositionVelocityState:
        """Rotate the state by the given angle in radiants"""
        rot_matrix = get_rotation_matrix(angle, self._states.device)
        rotated_pos = (rot_matrix @ self.position.unsqueeze(-1)).squeeze(-1)
        rotated_vel = (rot_matrix @ self.velocity.unsqueeze(-1)).squeeze(-1)
        if in_place:
            self._states = torch.cat((rotated_pos, rotated_vel), dim=-1)
            return self
        else:
            return to_state(torch.cat((rotated_pos, rotated_vel), dim=-1), self.dt)

    def translate(
        self, translation: torch.Tensor, in_place: bool = False
    ) -> PositionVelocityState:
        """Translate the state by the given tranlation"""
        if in_place:
            self._states[..., :2] += translation.expand_as(self._states[..., :2])
            return self
        else:
            return to_state(
                torch.cat(
                    (
                        self._states[..., :2].clone()
                        + translation.expand_as(self._states[..., :2]),
                        self._states[..., 2:].clone(),
                    ),
                    dim=-1,
                ),
                self.dt,
            )


class PositionAngleVelocityState(AbstractState):
    """
    State representation with an underlying tensor representing position angle and velocity.
    """

    def __init__(self, states: torch.Tensor, dt: float) -> None:
        super().__init__()
        assert states.shape[-1] == 5
        self._states = states.clone()
        self.dt = dt

    @property
    def position(self) -> torch.Tensor:
        return self._states[..., :2].clone()

    @property
    def velocity(self) -> torch.Tensor:
        return self._states[..., 3:5].clone()

    @property
    def angle(self) -> torch.Tensor:
        return self._states[..., 2:3].clone()

    def get_states(self, dim: int = 5) -> torch.Tensor:
        if dim == 2:
            return self._states[..., :2].clone()
        elif dim == 4:
            return torch.cat(
                (self._states[..., :2].clone(), self._states[..., 3:].clone()), dim=-1
            )
        elif dim == 5:
            return self._states.clone()
        else:
            raise RuntimeError(f"State dimension must be either 2, 4, or 5. Got {dim}")

    def rotate(
        self, angle: float, in_place: bool = False
    ) -> PositionAngleVelocityState:
        """Rotate the state by the given angle in radiants"""
        rot_matrix = get_rotation_matrix(angle, self._states.device)
        rotated_pos = (rot_matrix @ self.position.unsqueeze(-1)).squeeze(-1)
        rotated_angle = self.angle + angle
        rotated_vel = (rot_matrix @ self.velocity.unsqueeze(-1)).squeeze(-1)
        if in_place:
            self._states = torch.cat(rotated_pos, rotated_angle, rotated_vel, -1)
            return self
        else:
            return to_state(
                torch.cat(rotated_pos, rotated_angle, rotated_vel, -1), self.dt
            )

    def translate(
        self, translation: torch.Tensor, in_place: bool = False
    ) -> PositionAngleVelocityState:
        """Translate the state by the given tranlation"""
        if in_place:
            self._states[..., :2] += translation.expand_as(self._states[..., :2])
            return self
        else:
            return to_state(
                torch.cat(
                    (
                        self._states[..., :2]
                        + translation.expand_as(self._states[..., :2]),
                        self._states[..., 2:],
                    ),
                    dim=-1,
                ),
                self.dt,
            )


def get_interaction_cost(
    ego_state_future: AbstractState,
    ado_state_future_samples: AbstractState,
    interaction_cost_function: BaseCostTorch,
) -> torch.Tensor:
    """Computes interaction cost samples from predicted ado future trajectories and a batch of ego
    future trajectories

    Args:
        ego_state_future: ((num_control_samples), num_agents, num_steps_future) ego state future
            future trajectory
        ado_state_future_samples: (num_prediction_samples, num_agents, num_steps_future)
            predicted ado state trajectory samples
        interaction_cost_function: interaction cost function between ego and (stochastic) ado
        dt: time differential between two discrete timesteps in seconds

    Returns:
        (num_control_samples, num_agents, num_prediction_samples) interaction cost tensor
    """
    if len(ego_state_future.shape) == 2:
        y_ego = ego_state_future.position.unsqueeze(0)
        v_ego = ego_state_future.velocity.unsqueeze(0)
    else:
        y_ego = ego_state_future.position
        v_ego = ego_state_future.velocity

    num_control_samples = ego_state_future.shape[0]
    ado_position_future_samples = ado_state_future_samples.position.unsqueeze(0).expand(
        num_control_samples, -1, -1, -1, -1
    )

    v_samples = ado_state_future_samples.velocity.unsqueeze(0).expand(
        num_control_samples, -1, -1, -1, -1
    )

    interaction_cost, _ = interaction_cost_function(
        x1=y_ego.unsqueeze(1),
        x2=ado_position_future_samples,
        v1=v_ego.unsqueeze(1),
        v2=v_samples,
    )
    return interaction_cost.permute(0, 2, 1)


def evaluate_risk(
    risk_level: float,
    cost: torch.Tensor,
    weights: torch.Tensor,
    risk_estimator: Optional[AbstractMonteCarloRiskEstimator] = None,
) -> torch.Tensor:
    """Returns a risk tensor given costs and optionally a risk level

    Args:
        risk_level (optional): a risk-level float. If 0.0, risk-neutral expectation will be
          returned. Defaults to 0.0.
        cost: (num_control_samples, num_agents, num_prediction_samples) cost tensor
        weights: (num_control_samples, num_agents, num_prediction_samples) probability weight of the cost tensor
        risk_estimator (optional): a Monte Carlo risk estimator. Defaults to None.

    Returns:
        (num_control_samples, num_agents) risk tensor
    """
    num_control_samples, num_agents, _ = cost.shape

    if risk_level == 0.0:
        risk = cost.mean(dim=-1)
    else:
        assert risk_estimator is not None, "no risk estimator is specified"
        risk = risk_estimator(
            risk_level * torch.ones(num_control_samples, num_agents),
            cost,
            weights=weights,
        )
    return risk


def evaluate_control_sequence(
    control_sequence: torch.Tensor,
    dynamics_model,
    ego_state_history: AbstractState,
    ego_state_target_trajectory: AbstractState,
    ado_state_future_samples: AbstractState,
    sample_weights: torch.Tensor,
    interaction_cost_function: BaseCostTorch,
    tracking_cost_function: TrackingCost,
    risk_level: float = 0.0,
    risk_estimator: Optional[AbstractMonteCarloRiskEstimator] = None,
) -> Tuple[float, float]:
    """Returns the risk and tracking cost evaluation of the given control sequence

    Args:
        control_sequence: (num_steps_future, control_dim) tensor of control sequence
        dynamics_model: dynamics model for control
        ego_state_target_trajectory: (num_steps_future) tensor of ego target
          state trajectory
        ado_state_future_samples: (num_prediction_samples, num_agents, num_steps_future)
          of predicted ado trajectory samples states
        sample_weights: (num_prediction_samples, num_agents) tensor of probability weights of the samples
        intraction_cost_function: interaction cost function between ego and (stochastic) ado
        tracking_cost_function: deterministic tracking cost that does not involve ado
        risk_level: risk_level (optional): a risk-level float. If 0.0, risk-neutral expectation
          is used. Defaults to 0.0.
        risk_estimator (optional): a Monte Carlo risk estimator. Defaults to None.

    Returns:
        tuple of (interaction risk, tracking_cost)
    """
    ego_state_current = ego_state_history[..., -1]
    ego_state_future = dynamics_model.simulate(ego_state_current, control_sequence)
    # state starts with x, y, angle, vx, vy
    tracking_cost = tracking_cost_function(
        ego_state_future.position,
        ego_state_target_trajectory.position,
        ego_state_target_trajectory.velocity,
    )

    interaction_cost = get_interaction_cost(
        ego_state_future,
        ado_state_future_samples,
        interaction_cost_function,
    )

    interaction_risk = evaluate_risk(
        risk_level,
        interaction_cost,
        sample_weights.permute(1, 0).unsqueeze(0).expand_as(interaction_cost),
        risk_estimator,
    )

    # TODO: averaging over agents but we might want to reduce a different way
    return (interaction_risk.mean().item(), tracking_cost.mean().item())
