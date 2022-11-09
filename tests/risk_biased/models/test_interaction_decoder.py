import os
import pytest

import torch
from mmcv import Config

from risk_biased.models.cvae_decoder import (
    CVAEAccelerationDecoder,
    DecoderNN,
)
from risk_biased.models.cvae_params import CVAEParams


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    waymo_config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "waymo_config.py"
    )
    paths = [config_path, waymo_config_path]
    if isinstance(paths, str):
        cfg = Config.fromfile(paths)
    else:
        cfg = Config.fromfile(paths[0])
        for path in paths[1:]:
            c = Config.fromfile(path)
            cfg.update(c)
    cfg.batch_size = 4
    cfg.state_dim = 5
    cfg.map_state_dim = 2
    cfg.num_steps = 3
    cfg.num_steps_future = 4
    cfg.latent_dim = 2
    cfg.hidden_dim = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.device = "cpu"
    return cfg


@pytest.mark.parametrize(
    "num_agents, num_objects, n_samples, type",
    [
        (2, 3, 0, "MLP"),
        (3, 1, 2, "LSTM"),
        (4, 2, 2, "maskedLSTM"),
    ],
)
def test_interaction_decoder_nn(
    params, num_agents: int, num_objects: int, n_samples: int, type: str
):
    params.sequence_decoder_type = type
    model = DecoderNN(
        CVAEParams.from_config(params),
    )

    squeeze_sample_dim = n_samples <= 0
    n_samples = max(1, n_samples)
    x = torch.rand(params.batch_size, num_agents, params.num_steps, params.state_dim)
    mask_x = torch.rand(params.batch_size, num_agents, params.num_steps) > 0.3
    mask_z = mask_x.any(-1)
    z_samples = torch.rand(params.batch_size, num_agents, n_samples, params.latent_dim)
    encoded_map = torch.rand(params.batch_size, num_objects, params.hidden_dim)
    mask_map = torch.rand(params.batch_size, num_objects)
    encoded_absolute = torch.rand(params.batch_size, num_agents, params.hidden_dim)

    if squeeze_sample_dim:
        z_samples = z_samples.squeeze(2)

    output = model(
        z_samples, mask_z, x, mask_x, encoded_absolute, encoded_map, mask_map
    )

    # check shape
    if squeeze_sample_dim:
        assert output.shape == (
            params.batch_size,
            num_agents,
            params.num_steps_future,
            params.hidden_dim,
        )
    else:
        assert output.shape == (
            params.batch_size,
            num_agents,
            n_samples,
            params.num_steps_future,
            params.hidden_dim,
        )


@pytest.mark.parametrize(
    "num_agents, num_objects, n_samples, type",
    [
        (2, 3, 0, "MLP"),
        (3, 1, 2, "LSTM"),
        (4, 2, 2, "maskedLSTM"),
    ],
)
def test_interaction_cvae_decoder(
    params, num_agents: int, num_objects: int, n_samples: int, type: str
):
    params.sequence_decoder_type = type
    squeeze_sample_dim = n_samples <= 0
    n_samples = max(1, n_samples)
    z_samples = torch.rand(params.batch_size, num_agents, n_samples, params.latent_dim)
    if squeeze_sample_dim == 1:
        z_samples = z_samples.squeeze(2)
    x = torch.rand(params.batch_size, num_agents, params.num_steps, params.state_dim)
    offset = torch.rand(params.batch_size, num_agents, 5)
    mask_x = torch.rand(params.batch_size, num_agents, params.num_steps) > 0.3
    mask_z = mask_x.any(-1)
    encoded_map = torch.rand(params.batch_size, num_objects, params.hidden_dim)
    mask_map = torch.rand(params.batch_size, num_objects)
    encoded_absolute = torch.rand(params.batch_size, num_agents, params.hidden_dim)

    model = DecoderNN(CVAEParams.from_config(params))
    decoder = CVAEAccelerationDecoder(model)
    # check auxiliary_input_dim
    y_samples = decoder(
        z_samples,
        mask_z,
        x,
        mask_x,
        encoded_absolute,
        encoded_map,
        mask_map,
        offset=offset,
    )
    # check shape
    if squeeze_sample_dim:
        assert y_samples.shape == (
            params.batch_size,
            num_agents,
            params.num_steps_future,
            params.state_dim,
        )
    else:
        assert y_samples.shape == (
            params.batch_size,
            num_agents,
            n_samples,
            params.num_steps_future,
            params.state_dim,
        )
