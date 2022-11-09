import os
import pytest
import torch
import numpy as np
from mmcv import Config

from risk_biased.utils.cost import BaseCostTorch, TTCCostTorch, DistanceCostTorch
from risk_biased.utils.cost import BaseCostNumpy, TTCCostNumpy, DistanceCostNumpy
from risk_biased.utils.cost import (
    CostParams,
    TTCCostParams,
    DistanceCostParams,
)


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = Config.fromfile(config_path)

    cfg.cost_scale = 1
    cfg.cost_reduce = "mean"
    cfg.ego_length = 4
    cfg.ego_width = 1.75
    cfg.distance_bandwidth = 2
    cfg.time_bandwidth = 2
    cfg.min_velocity_diff = 0.01

    return cfg


def get_fake_input(batch_size, num_steps, is_torch, use_mask, num_agents=0):
    if num_agents <= 0:
        shape = [batch_size, num_steps, 2]
    else:
        shape = [batch_size, num_agents, num_steps, 2]
    if is_torch:
        x1 = torch.rand(shape)
        x2 = torch.rand(shape)
        v1 = torch.rand(shape)
        v2 = torch.rand(shape)
        if use_mask:
            mask = torch.rand(shape[:-1]) > 0.1
        else:
            mask = None
    else:
        x1 = np.random.uniform(size=shape)
        x2 = np.random.uniform(size=shape)
        v1 = np.random.uniform(size=shape)
        v2 = np.random.uniform(size=shape)
        if use_mask:
            mask = np.random.uniform(size=shape[:-1]) > 0.1
        else:
            mask = None
    return x1, x2, v1, v2, mask


@pytest.mark.parametrize(
    "reduce, batch_size, num_steps, is_torch, use_mask, num_agents",
    [
        ("mean", 8, 5, True, True, 0),
        ("min", 4, 2, False, True, 2),
        ("max", 4, 2, True, False, 3),
        ("now", 16, 1, False, False, 1),
        ("final", 1, 4, True, True, 0),
    ],
)
def test_base_cost(
    params,
    reduce: str,
    batch_size: int,
    num_steps: int,
    is_torch: bool,
    use_mask: bool,
    num_agents: int,
):
    params.cost_reduce = reduce
    cost_params = CostParams.from_config(params)
    if is_torch:
        base_cost = BaseCostTorch(cost_params)
    else:
        base_cost = BaseCostNumpy(cost_params)

    x1, x2, v1, v2, mask = get_fake_input(
        batch_size, num_steps, is_torch, use_mask, num_agents
    )
    cost, _ = base_cost(x1, x2, v1, v2, mask)
    if num_agents > 0:
        assert cost.shape == (
            batch_size,
            num_agents,
        )
    else:
        assert cost.shape == (batch_size,)
    assert (cost == 0).all()
    assert base_cost.scale == params.cost_scale
    assert base_cost.distance_bandwidth == 1
    assert base_cost.time_bandwidth == 1


@pytest.mark.parametrize(
    "param_class, cost_class, reduce, batch_size, num_steps, is_torch, use_mask, num_agents",
    [
        (DistanceCostParams, DistanceCostTorch, "max", 4, 2, True, True, 3),
        (DistanceCostParams, DistanceCostNumpy, "now", 16, 1, False, True, 0),
        (DistanceCostParams, DistanceCostTorch, "final", 1, 4, True, False, 2),
        (TTCCostParams, TTCCostTorch, "max", 4, 2, True, False, 0),
        (TTCCostParams, TTCCostNumpy, "now", 16, 1, False, True, 3),
        (TTCCostParams, TTCCostNumpy, "final", 1, 4, False, True, 1),
    ],
)
def test_generic_cost(
    params,
    param_class,
    cost_class,
    reduce: str,
    batch_size: int,
    num_steps: int,
    is_torch: bool,
    use_mask: bool,
    num_agents: int,
):
    params.cost_reduce = reduce
    cost_params = param_class.from_config(params)
    x1, x2, v1, v2, mask = get_fake_input(
        batch_size, num_steps, is_torch, use_mask, num_agents
    )

    compute_cost = cost_class(cost_params)

    cost, _ = compute_cost(x1, x2, v1, v2, mask)
    # Shaped is reduced
    if num_agents > 0:
        assert cost.shape == (batch_size, num_agents)
    else:
        assert cost.shape == (batch_size,)
    assert (cost != 0).any()
    assert compute_cost.scale == params.cost_scale
    # Rescale the cost for comparison
    compute_cost.scale = params.cost_scale + 10
    assert compute_cost.scale != params.cost_scale
    rescaled_cost, _ = compute_cost(x1, x2, v1, v2, mask)
    # all rescaled cost are larger but 0 cost is equal to rescaled cost
    assert (rescaled_cost >= cost).all()
    # at least some rescaled cost are strictly larger than normal scale cost
    assert (rescaled_cost > cost).any()

    # Compute mean and min costs to compare
    params.cost_reduce = "mean"
    cost_params_mean = param_class.from_config(params)
    cost_function_mean = cost_class(cost_params_mean)
    cost_mean, _ = cost_function_mean(x1, x2, v1, v2)

    params.cost_reduce = "min"
    cost_params_min = param_class.from_config(params)
    cost_function_min = cost_class(cost_params_min)
    cost_min, _ = cost_function_min(x1, x2, v1, v2)

    # max reduce is larger than mean
    if reduce == "max":
        assert (cost >= cost_mean).all()
    # min reduce is lower than any othir
    assert (cost_mean >= cost_min).all()
    assert (cost >= cost_min).all()
