import torch
import torch.nn as nn
from risk_biased.models.nn_blocks import (
    SequenceEncoderLSTM,
    SequenceEncoderMLP,
    SequenceEncoderMaskedLSTM,
)

from risk_biased.models.cvae_params import CVAEParams
from risk_biased.models.mlp import MLP


class MapEncoderNN(nn.Module):
    """MLP encoder neural network that encodes map objects.

    Args:
        params: dataclass defining the necessary parameters
    """

    def __init__(self, params: CVAEParams) -> None:
        super().__init__()
        self._encoder = SequenceEncoderMLP(
            params.map_state_dim,
            params.hidden_dim,
            params.num_hidden_layers,
            params.max_size_lane,
            params.is_mlp_residual,
        )

    def forward(self, map, mask_map):
        """Forward function encoding map object sequences of features into object features.

        Args:
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor of bool mask
        """
        encoded_map = self._encoder(map, mask_map)
        return encoded_map
