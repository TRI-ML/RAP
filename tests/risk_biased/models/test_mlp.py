import pytest

import torch
from mmcv import Config

from risk_biased.models.mlp import MLP


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    cfg = Config()
    cfg.batch_size = 4
    cfg.input_dim = 10
    cfg.output_dim = 15
    cfg.latent_dim = 3
    cfg.h_dim = 64
    cfg.num_h_layers = 2
    cfg.device = "cpu"
    cfg.is_mlp_residual = True
    return cfg


def test_mlp(params):
    mlp = MLP(
        params.input_dim,
        params.output_dim,
        params.h_dim,
        params.num_h_layers,
        params.is_mlp_residual,
    )

    input = torch.rand(params.batch_size, params.input_dim)
    output = mlp(input)
    # check shape
    assert output.shape == (params.batch_size, params.output_dim)
