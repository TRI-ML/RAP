import pytest

import torch

from risk_biased.mpc_planner.dynamics import PositionVelocityDoubleIntegrator
from risk_biased.utils.planner_utils import to_state


@pytest.mark.parametrize("dt", [(0.01), (0.1)])
def test_double_integrator(dt: float):

    torch.manual_seed(0)

    dynamics = PositionVelocityDoubleIntegrator(dt)
    assert dynamics.dt == dt
    assert dynamics.control_dim == 2

    state_init = to_state(torch.randn(1, 4), dt)
    control_input = torch.randn(10, 5, 2)
    state_future = dynamics.simulate(state_init, control_input)
    assert state_future.shape == (10, 5)

    assert torch.allclose(
        state_future.position,
        state_init.position
        + torch.cumsum(
            state_init.velocity + torch.cumsum(control_input, dim=1) * dt, dim=1
        )
        * dt,
    )
    assert torch.allclose(
        state_future.position,
        state_init.position + torch.cumsum(state_future.velocity * dt, dim=1),
    )
