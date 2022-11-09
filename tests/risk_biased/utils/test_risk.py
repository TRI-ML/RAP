import math
import pytest

import torch

from risk_biased.utils.risk import (
    CVaREstimator,
    EntropicRiskEstimator,
    get_risk_estimator,
    get_risk_level_sampler,
)

torch.manual_seed(0)


@pytest.mark.parametrize(
    "estimator_params, batch_size, num_samples",
    [
        ({"type": "entropic", "eps": 1e-3}, 3, 10),
        ({"type": "entropic", "eps": 1e-6}, 5, 20),
    ],
)
def test_entropic_risk_estimator(
    estimator_params: dict, batch_size: int, num_samples: int
):
    num_agents = 1
    estimator = get_risk_estimator(estimator_params)
    assert type(estimator) == EntropicRiskEstimator

    cost = torch.rand(batch_size, num_agents, num_samples)
    risk_level_random = torch.rand(batch_size, num_agents)
    weight = torch.ones(batch_size, num_agents, num_samples) / num_samples
    objective_random = estimator(risk_level_random, cost, weight)
    assert objective_random.shape == torch.Size([batch_size, num_agents])

    risk_level_zero = torch.zeros(batch_size, num_agents)
    objective_zero = estimator(risk_level_zero, cost, weight)
    # entropic risk should fall back to mean if risk_level is zero
    assert torch.allclose(objective_zero, (cost * weight).sum(dim=2))

    cost_same = torch.ones(batch_size, num_agents, num_samples)
    objective_same = estimator(risk_level_random, cost_same, weight)
    # entropic risk should return mean if cost samples are all the same
    assert torch.allclose(objective_same, (cost_same * weight).sum(dim=2))

    risk_level_one = torch.ones(batch_size, num_agents)
    objective_one = estimator(risk_level_one, cost, weight)
    risk_level_nine = 10.0 * torch.ones(batch_size, num_agents)
    objective_ten = estimator(risk_level_nine, cost, weight)
    # entropic risk should be monotone increasing as a function of risk_level
    assert all(objective_ten > objective_one)


@pytest.mark.parametrize(
    "estimator_params, batch_size, num_samples, num_agents",
    [
        ({"type": "cvar", "eps": 1e-3}, 3, 10, 1),
        ({"type": "cvar", "eps": 1e-6}, 5, 20, 3),
    ],
)
def test_cvar_estimator(
    estimator_params: dict, batch_size: int, num_samples: int, num_agents: int
):
    estimator = get_risk_estimator(estimator_params)
    assert type(estimator) == CVaREstimator

    cost = torch.rand(batch_size, num_agents, num_samples)
    risk_level_random = torch.rand(batch_size, num_agents)
    weights = torch.ones(batch_size, num_agents, num_samples) / num_samples
    objective_random = estimator(risk_level_random, cost, weights)
    assert objective_random.shape == torch.Size([batch_size, num_agents])

    risk_level_zero = torch.zeros(batch_size, num_agents)
    objective_zero = estimator(risk_level_zero, cost, weights)
    # cvar should fall back to mean if risk_level is zero
    assert torch.allclose(objective_zero, cost.mean(dim=2), rtol=1e-3, atol=1e-3)

    cost_same = torch.ones(batch_size, num_agents, num_samples)
    objective_same = estimator(risk_level_random, cost_same, weights)
    # cvar should return mean if cost samples are all the same
    assert torch.allclose(objective_same, cost_same.mean(dim=2))

    risk_level_close_to_one = torch.ones(batch_size, num_agents) - 1e-2
    objective_close_to_one = estimator(risk_level_close_to_one, cost, weights)
    risk_level_one = torch.ones(batch_size, num_agents)
    objective_one = estimator(risk_level_one, cost, weights)
    # cvar should fall back to max if risk_level is close to one
    assert torch.allclose(objective_close_to_one, cost.max(dim=2).values)
    assert torch.allclose(objective_one, cost.max(dim=2).values)

    risk_level_quarter = 0.25 * torch.ones(batch_size, num_agents)
    objective_quarter = estimator(risk_level_quarter, cost, weights)
    risk_level_half = 0.5 * torch.ones(batch_size, num_agents)
    objective_half = estimator(risk_level_half, cost, weights)
    # cvar should be monotone increasing as a function of risk_level
    assert (objective_half > objective_quarter).all()


def test_risk_estimator_raise():
    with pytest.raises(RuntimeError):
        get_risk_estimator({})
    with pytest.raises(RuntimeError):
        get_risk_estimator({"type": "entropic"})
    with pytest.raises(RuntimeError):
        get_risk_estimator({"eps": 1e-3})


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        ({"type": "uniform", "min": 0, "max": 1}, 1000, "cpu"),
        ({"type": "uniform", "min": 0, "max": 1}, 10000, "cuda"),
        ({"type": "uniform", "min": 10, "max": 100}, 10000, "cpu"),
    ],
)
def test_uniform_sampler(distribution_params: dict, num_samples: int, device: str):
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 2 / 5)

    expected_mean = (distribution_params["max"] + distribution_params["min"]) / 2
    expected_std = (
        distribution_params["max"] - distribution_params["min"]
    ) / math.sqrt(12)
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        ({"type": "normal", "mean": 0, "sigma": 1}, 1000, "cpu"),
        ({"type": "normal", "mean": 0, "sigma": 3}, 10000, "cuda"),
        ({"type": "normal", "mean": 3, "sigma": 10}, 10000, "cpu"),
        ({"type": "normal", "mean": 1, "sigma": 3}, 100000, "cpu"),
    ],
)
def test_normal_sampler(distribution_params: dict, num_samples: int, device: str):
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 2 / 5)
    expected_mean = distribution_params["mean"]
    expected_std = distribution_params["sigma"]
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        ({"type": "bernoulli", "min": 0, "max": 1, "p": 0.5}, 1000, "cpu"),
        ({"type": "bernoulli", "min": 0, "max": 3, "p": 0.1}, 10000, "cuda"),
        ({"type": "bernoulli", "min": 3, "max": 10, "p": 0.9}, 10000, "cpu"),
        ({"type": "bernoulli", "min": 1, "max": 3, "p": 0.5}, 100000, "cpu"),
    ],
)
def test_bernoulli_sampler(distribution_params: dict, num_samples: int, device: str):
    range = distribution_params["max"] - distribution_params["min"]
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 2 / 5)
    expected_mean = distribution_params["p"] * range + distribution_params["min"]
    expected_std = (
        math.sqrt(distribution_params["p"] * (1 - distribution_params["p"])) * range
    )
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        ({"type": "beta", "min": 0, "max": 1, "alpha": 0.5, "beta": 0.5}, 1000, "cpu"),
        ({"type": "beta", "min": 0, "max": 3, "alpha": 5, "beta": 1}, 10000, "cuda"),
        ({"type": "beta", "min": 3, "max": 10, "alpha": 1, "beta": 3}, 10000, "cpu"),
        ({"type": "beta", "min": 1, "max": 3, "alpha": 2, "beta": 5}, 100000, "cpu"),
    ],
)
def test_beta_sampler(distribution_params: dict, num_samples: int, device: str):
    range = distribution_params["max"] - distribution_params["min"]
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 2 / 5)
    aphaplusbeta = distribution_params["alpha"] + distribution_params["beta"]
    expected_mean = (
        distribution_params["alpha"] / aphaplusbeta * range + distribution_params["min"]
    )
    expected_std = (
        math.sqrt(distribution_params["alpha"] * distribution_params["beta"])
        / (aphaplusbeta * math.sqrt(aphaplusbeta + 1))
        * range
    )
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        ({"type": "chi2", "min": 0, "scale": 1, "k": 1}, 1000, "cpu"),
        ({"type": "chi2", "min": 0, "scale": 3, "k": 2}, 10000, "cuda"),
        ({"type": "chi2", "min": 3, "scale": 10, "k": 3}, 10000, "cpu"),
        ({"type": "chi2", "min": 1, "scale": 3, "k": 10}, 100000, "cpu"),
    ],
)
def test_chi2_sampler(distribution_params: dict, num_samples: int, device: str):

    tol_mean = (
        3
        * distribution_params["scale"]
        * math.sqrt(2 * distribution_params["k"] / num_samples)
    )
    tol_std = 3 / math.pow(num_samples, 2 / 5)
    expected_mean = (
        distribution_params["k"] * distribution_params["scale"]
        + distribution_params["min"]
    )
    expected_std = (
        math.sqrt(2 * distribution_params["k"]) * distribution_params["scale"]
    )
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        (
            {"type": "log-normal", "min": 0, "scale": 1, "mu": 0, "sigma": 0.5},
            100,
            "cpu",
        ),
        (
            {"type": "log-normal", "min": 0, "scale": 3, "mu": 0.3, "sigma": 1},
            100000,
            "cuda",
        ),
        (
            {"type": "log-normal", "min": 3, "scale": 10, "mu": 1, "sigma": 0.25},
            10000,
            "cpu",
        ),
        (
            {"type": "log-normal", "min": 1, "scale": 3, "mu": 0, "sigma": 1.5},
            100000,
            "cpu",
        ),
    ],
)
def test_lognormal_sampler(distribution_params: dict, num_samples: int, device: str):
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 1 / 5)
    expected_mean = (
        math.exp(distribution_params["mu"] + distribution_params["sigma"] ** 2 / 2)
        * distribution_params["scale"]
        + distribution_params["min"]
    )
    expected_std = (
        math.sqrt(math.exp(distribution_params["sigma"] ** 2) - 1)
        * math.exp(distribution_params["mu"] + (distribution_params["sigma"] ** 2) / 2)
    ) * distribution_params["scale"]
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


@pytest.mark.parametrize(
    "distribution_params, num_samples, device",
    [
        (
            {"type": "log-uniform", "min": 0, "max": 1, "scale": 1},
            100,
            "cpu",
        ),
        (
            {"type": "log-uniform", "min": 1, "max": 3, "scale": 3},
            100000,
            "cuda",
        ),
    ],
)
def test_loguniform_sampler(distribution_params: dict, num_samples: int, device: str):
    tol_mean = 3 / math.sqrt(num_samples)
    tol_std = 3 / math.pow(num_samples, 1 / 5)
    max = distribution_params["max"]
    min = distribution_params["min"]
    scale = distribution_params["scale"] / (max - min)
    expected_mean = (
        max
        - ((max - min) / math.log((scale * max + 1) / (scale * min + 1)) - 1 / scale)
        + min
    )
    expected_std = math.sqrt(
        ((scale * max + 1) ** 2 - (scale * min + 1) ** 2)
        / (2 * scale**2 * math.log((scale * max + 1) / (scale * min + 1)))
        - ((max - min) / math.log((scale * max + 1) / (scale * min + 1))) ** 2
    )
    sampler = get_risk_level_sampler(distribution_params=distribution_params)
    sample = sampler.sample(num_samples, device)
    std, mean = torch.std_mean(sample)
    assert sample.shape == torch.Size([num_samples])
    assert torch.abs(mean - expected_mean) / expected_std < tol_mean
    assert torch.abs(std - expected_std) / expected_std < tol_std


def test_risk_level_sampler_raise():
    with pytest.raises(RuntimeError):
        get_risk_level_sampler({})
    with pytest.raises(RuntimeError):
        get_risk_level_sampler({"type": "chi2"})
    with pytest.raises(RuntimeError):
        get_risk_level_sampler({"min": 0, "max": 1})
