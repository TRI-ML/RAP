import torch

from risk_biased.utils.planner_utils import AbstractState, to_state


class PositionVelocityDoubleIntegrator:
    """Deterministic discrete-time double-integrator dynamics, where state is
    [position_x_m, position_y_m, velocity_x_m_s velocity_y_m_s] and control is
    [acceleration_x_m_s2, acceleration_y_m_s2].

    Args:
        dt: time differential between two discrete timesteps in seconds
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.control_dim = 2

    def simulate(
        self,
        state_init: AbstractState,
        control_input: torch.Tensor,
    ) -> AbstractState:
        """Euler-integrate dynamics from the initial position and the initial velocity given
        an acceleration input

        Args:
            state_init: (some_shape) initial Markov state of the system
            control_input: (some_shape, num_steps_future, 2) tensor of acceleration input

        Returns:
            (some_shape, num_steps_future, 5) tensor of simulated future Markov state
              sequence
        """
        position_init, velocity_init = state_init.position, state_init.velocity

        assert (
            control_input.shape[-1] == self.control_dim
        ), "invalid control input dimension"

        velocity_future = velocity_init + self.dt * torch.cumsum(control_input, dim=-2)

        position_future = position_init + self.dt * torch.cumsum(
            velocity_future, dim=-2
        )
        state_future = to_state(
            torch.cat((position_future, velocity_future), dim=-1), self.dt
        )
        return state_future
