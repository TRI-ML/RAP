from collections import OrderedDict

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Basic MLP implementation with FC layers and ReLU activation.

    Args:
        input_dim : dimension of the input variable
        output_dim : dimension of the output variable
        h_dim : dimension of a hidden layer of MLP
        num_h_layers : number of hidden layers in MLP
        add_residual : set to True to add input to output (res-net) set to False to have pure MLP

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        h_dim: int,
        num_h_layers: int,
        add_residual: bool,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._h_dim = h_dim
        self._num_h_layers = num_h_layers

        layers = OrderedDict()
        if num_h_layers > 0:
            layers["fc_0"] = nn.Linear(input_dim, h_dim)
            layers["relu_0"] = nn.ReLU()
        else:
            h_dim = input_dim
        for ii in range(1, num_h_layers):
            layers["fc_{}".format(ii)] = nn.Linear(h_dim, h_dim)
            layers["relu_{}".format(ii)] = nn.ReLU()
        layers["fc_{}".format(num_h_layers)] = nn.Linear(h_dim, self._output_dim)

        self.mlp = nn.Sequential(layers)
        if add_residual:
            self.residual_layer = nn.Linear(input_dim, output_dim)
        else:
            self.residual_layer = lambda x: 0
        self._layer_norm = nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward function for MLP

        Args:
            input (torch.Tensor): (batch_size, input_dim) tensor

        Returns:
            torch.Tensor: (batch_size, output_dim) tensor
        """
        return self._layer_norm(self.mlp(input) + self.residual_layer(input))
