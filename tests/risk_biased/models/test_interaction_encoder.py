import os
import pytest

import torch
import torch.nn as nn
from mmcv import Config

from risk_biased.models.cvae_encoders import (
    CVAEEncoder,
    BiasedEncoderNN,
    FutureEncoderNN,
    InferenceEncoderNN,
)
from risk_biased.models.latent_distributions import (
    GaussianLatentDistribution,
    QuantizedDistributionCreator,
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
    cfg.dynamic_state_dim = 5
    cfg.map_state_dim = 2
    cfg.num_steps = 3
    cfg.num_steps_future = 4
    cfg.latent_dim = 2
    cfg.hidden_dim = 64
    cfg.device = "cpu"
    cfg.sequence_encoder_type = "LSTM"
    cfg.sequence_decoder_type = "MLP"
    return cfg


@pytest.mark.parametrize(
    "num_agents, num_map_objects, type, interaction_nn_class",
    [
        (4, 5, "MLP", BiasedEncoderNN),
        (2, 4, "LSTM", BiasedEncoderNN),
        (3, 2, "maskedLSTM", BiasedEncoderNN),
        (4, 5, "MLP", FutureEncoderNN),
        (2, 4, "LSTM", FutureEncoderNN),
        (3, 2, "maskedLSTM", FutureEncoderNN),
        (4, 5, "MLP", InferenceEncoderNN),
        (2, 4, "LSTM", InferenceEncoderNN),
        (3, 2, "maskedLSTM", InferenceEncoderNN),
    ],
)
def test_attention_encoder_nn(
    params,
    num_agents: int,
    num_map_objects: int,
    type: str,
    interaction_nn_class: nn.Module,
):
    params.sequence_encoder_type = type
    cvae_params = CVAEParams.from_config(params)
    if interaction_nn_class == BiasedEncoderNN:
        model = interaction_nn_class(
            cvae_params,
            num_steps=cvae_params.num_steps,
            latent_dim=2 * cvae_params.latent_dim,
        )
    elif interaction_nn_class == FutureEncoderNN:
        model = interaction_nn_class(
            cvae_params,
            num_steps=cvae_params.num_steps + cvae_params.num_steps_future,
            latent_dim=2 * cvae_params.latent_dim,
        )
    else:
        model = interaction_nn_class(
            cvae_params,
            num_steps=cvae_params.num_steps,
            latent_dim=2 * cvae_params.latent_dim,
        )
    assert model.latent_dim == 2 * params.latent_dim
    assert model.hidden_dim == params.hidden_dim

    x = torch.rand(params.batch_size, num_agents, params.num_steps, params.state_dim)
    offset = x[:, :, -1, :]
    x = x - offset.unsqueeze(-2)
    mask_x = torch.rand(params.batch_size, num_agents, params.num_steps) > 0.1
    encoded_absolute = torch.rand(params.batch_size, num_agents, params.hidden_dim)
    encoded_map = torch.rand(params.batch_size, num_map_objects, params.hidden_dim)
    mask_map = torch.rand(params.batch_size, num_map_objects) > 0.1
    if interaction_nn_class == FutureEncoderNN:
        y = torch.rand(
            params.batch_size, num_agents, params.num_steps_future, params.state_dim
        )
        y = y - offset.unsqueeze(-2)
        y_ego = y[:, 0:1]
        mask_y = torch.rand(params.batch_size, num_agents, params.num_steps_future)
    else:
        y = None
        y_ego = None
        mask_y = None
    x_ego = x[:, 0:1]
    if interaction_nn_class == BiasedEncoderNN:
        risk_level = torch.rand(params.batch_size, num_agents)
    else:
        risk_level = None

    output = model(
        x,
        mask_x,
        encoded_absolute,
        encoded_map,
        mask_map,
        y=y,
        mask_y=mask_y,
        x_ego=x_ego,
        y_ego=y_ego,
        offset=offset,
        risk_level=risk_level,
    )
    # check shape
    assert output.shape == (params.batch_size, num_agents, 2 * params.latent_dim)


@pytest.mark.parametrize(
    "num_agents, num_map_objects, type, interaction_nn_class, latent_distribution_class",
    [
        (2, 8, "MLP", BiasedEncoderNN, GaussianLatentDistribution),
        (7, 5, "LSTM", BiasedEncoderNN, GaussianLatentDistribution),
        (2, 10, "maskedLSTM", BiasedEncoderNN, QuantizedDistributionCreator),
        (2, 8, "MLP", FutureEncoderNN, GaussianLatentDistribution),
        (7, 5, "LSTM", FutureEncoderNN, QuantizedDistributionCreator),
        (2, 10, "maskedLSTM", FutureEncoderNN, GaussianLatentDistribution),
        (2, 8, "MLP", InferenceEncoderNN, QuantizedDistributionCreator),
        (7, 5, "LSTM", InferenceEncoderNN, GaussianLatentDistribution),
        (2, 10, "maskedLSTM", InferenceEncoderNN, GaussianLatentDistribution),
    ],
)
# TODO: Add test for QuantizedDistributionCreator
def test_attention_cvae_encoder(
    params,
    num_agents: int,
    num_map_objects: int,
    type: str,
    interaction_nn_class,
    latent_distribution_class,
):
    params.sequence_encoder_type = type
    if interaction_nn_class == FutureEncoderNN:
        risk_level = None
        y = torch.rand(
            params.batch_size, num_agents, params.num_steps_future, params.state_dim
        )
        mask_y = torch.rand(params.batch_size, num_agents, params.num_steps_future)
    else:
        risk_level = torch.rand(params.batch_size, num_agents)
        y = None
        mask_y = None

    if interaction_nn_class == BiasedEncoderNN:
        model = interaction_nn_class(
            CVAEParams.from_config(params),
            num_steps=params.num_steps,
            latent_dim=2 * params.latent_dim,
        )
    elif interaction_nn_class == FutureEncoderNN:
        model = interaction_nn_class(
            CVAEParams.from_config(params),
            num_steps=params.num_steps + params.num_steps_future,
            latent_dim=2 * params.latent_dim,
        )
    else:
        model = interaction_nn_class(
            CVAEParams.from_config(params),
            num_steps=params.num_steps,
            latent_dim=2 * params.latent_dim,
        )

    encoder = CVAEEncoder(model, GaussianLatentDistribution)
    # check latent_dim
    assert encoder.latent_dim == 2 * params.latent_dim

    x = torch.rand(params.batch_size, num_agents, params.num_steps, params.state_dim)
    offset = x[:, :, -1, :]
    x = x - offset.unsqueeze(-2)
    if y is not None:
        y = y - offset.unsqueeze(-2)
        x_ego = x[:, 0:1]
        y_ego = y[:, 0:1]
    else:
        x_ego = x[:, 0:1]
        y_ego = None
    mask_x = torch.rand(params.batch_size, num_agents, params.num_steps) > 0.1
    encoded_absolute = torch.rand(params.batch_size, num_agents, params.hidden_dim)
    encoded_map = torch.rand(params.batch_size, num_map_objects, params.hidden_dim)
    mask_map = torch.rand(params.batch_size, num_map_objects) > 0.1

    latent_distribution = encoder(
        x=x,
        mask_x=mask_x,
        encoded_absolute=encoded_absolute,
        encoded_map=encoded_map,
        mask_map=mask_map,
        y=y,
        mask_y=mask_y,
        x_ego=x_ego,
        y_ego=y_ego,
        offset=offset,
        risk_level=risk_level,
    )
    latent_mean = latent_distribution.mu
    latent_log_std = latent_distribution.logvar
    # check shape
    assert (
        latent_mean.shape
        == latent_log_std.shape
        == (params.batch_size, num_agents, params.latent_dim)
    )

    latent_sample_1, weights = latent_distribution.sample()
    # check shape when n_samples = 0
    assert latent_sample_1.shape == latent_mean.shape
    assert latent_sample_1.shape[:-1] == weights.shape

    latent_sample_2, weights = latent_distribution.sample(n_samples=2)
    # check shape when n_samples = 2
    assert latent_sample_2.shape == (
        params.batch_size,
        num_agents,
        2,
        params.latent_dim,
    )

    latent_sample_3, weights = latent_distribution.sample()
    # make sure sampling is non-deterministic
    assert not torch.allclose(latent_sample_1, latent_sample_3)
