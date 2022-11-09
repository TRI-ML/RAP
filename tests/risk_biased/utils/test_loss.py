import os
import pytest

import torch
from mmcv import Config

from risk_biased.models.latent_distributions import GaussianLatentDistribution


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = Config.fromfile(config_path)
    cfg.batch_size = 4
    cfg.latent_dim = 2
    return cfg


@pytest.mark.parametrize("threshold", [(1e-5), (10.0)])
def test_get_kl_loss(params, threshold: float):
    z_mean_log_std = torch.rand(params.batch_size, 1, params.latent_dim * 2)

    distribution = GaussianLatentDistribution(z_mean_log_std)

    z_mean, z_log_var = torch.split(z_mean_log_std, params.latent_dim, dim=-1)
    z_log_std = z_log_var / 2.0

    kl_target = (
        (
            -0.5 * (1.0 + 2.0 * z_log_std - z_mean.square() - (2 * z_log_std).exp())
        ).clamp_min(threshold)
    ).mean()

    prior_z_mean_log_std = torch.zeros(params.latent_dim * 2)
    prior_distribution = GaussianLatentDistribution(prior_z_mean_log_std)

    # Test kl loss is 0 on identical distributions
    assert torch.isclose(
        distribution.kl_loss(distribution, threshold=threshold),
        torch.zeros(1),
        atol=threshold,
    )

    # test kl loss when prior is unit Gaussian
    assert torch.isclose(
        distribution.kl_loss(prior_distribution, threshold),
        kl_target,
    )
