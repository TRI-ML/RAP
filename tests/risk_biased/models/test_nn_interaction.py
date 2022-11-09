from cmath import isnan
import pytest

import torch
from mmcv import Config

from risk_biased.models.nn_blocks import (
    SequenceDecoderLSTM,
    SequenceDecoderMLP,
    SequenceEncoderMaskedLSTM,
    SequenceEncoderMLP,
    AttentionBlock,
)


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    cfg = Config()
    cfg.batch_size = 4
    cfg.input_dim = 10
    cfg.output_dim = 15
    cfg.latent_dim = 3
    cfg.h_dim = 32
    cfg.num_attention_heads = 4
    cfg.num_h_layers = 2
    cfg.device = "cpu"
    return cfg


def test_AttentionBlock(params):
    attention = AttentionBlock(params.h_dim, params.num_attention_heads)
    num_agents = 4
    num_map_objects = 8
    encoded_agents = torch.rand(params.batch_size, num_agents, params.h_dim)
    mask_agents = torch.rand(params.batch_size, num_agents) > 0.1
    encoded_absolute_agents = torch.rand(params.batch_size, num_agents, params.h_dim)
    encoded_map = torch.rand(params.batch_size, num_map_objects, params.h_dim)
    mask_map = torch.rand(params.batch_size, num_map_objects) > 0.1
    output = attention(
        encoded_agents, mask_agents, encoded_absolute_agents, encoded_map, mask_map
    )
    # check shape
    assert output.shape == (params.batch_size, num_agents, params.h_dim)
    assert not torch.isnan(output).any()


def test_SequenceDecoder(params):
    decoder = SequenceDecoderLSTM(params.h_dim)
    num_agents = 8
    sequence_length = 16

    input = torch.rand(params.batch_size, num_agents, params.h_dim)

    output = decoder(input, sequence_length)

    assert output.shape == (
        params.batch_size,
        num_agents,
        sequence_length,
        params.h_dim,
    )
    assert not torch.isnan(output).any()


def test_SequenceDecoderMLP(params):
    sequence_length = 16
    decoder = SequenceDecoderMLP(
        params.h_dim, params.num_h_layers, sequence_length, True
    )
    num_agents = 8

    input = torch.rand(params.batch_size, num_agents, params.h_dim)

    output = decoder(input, sequence_length)

    assert output.shape == (
        params.batch_size,
        num_agents,
        sequence_length,
        params.h_dim,
    )
    assert not torch.isnan(output).any()


def test_SequenceEncoder(params):
    encoder = SequenceEncoderMaskedLSTM(params.input_dim, params.h_dim)
    num_agents = 8
    sequence_length = 16

    input = torch.rand(params.batch_size, num_agents, sequence_length, params.input_dim)
    mask_input = torch.rand(params.batch_size, num_agents, sequence_length) > 0.1

    output = encoder(input, mask_input)

    assert output.shape == (params.batch_size, num_agents, params.h_dim)
    assert not torch.isnan(output).any()


def test_SequenceEncoderMLP(params):
    sequence_length = 16
    num_agents = 8
    encoder = SequenceEncoderMLP(
        params.input_dim, params.h_dim, params.num_h_layers, sequence_length, True
    )

    input = torch.rand(params.batch_size, num_agents, sequence_length, params.input_dim)
    mask_input = torch.rand(params.batch_size, num_agents, sequence_length) > 0.1

    output = encoder(input, mask_input)

    assert output.shape == (params.batch_size, num_agents, params.h_dim)
    assert not torch.isnan(output).any()
