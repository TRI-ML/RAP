import torch
from torch import nn
from risk_biased.models.mlp import MLP


def pool(x, dim):
    x, _ = x.max(dim)
    return x


class ContextGating(nn.Module):
    """Inspired by Multi-Path++ https://arxiv.org/pdf/2111.14973v3.pdf (but not the same)

    Args:
        d_model: input dimension of the model
        d: hidden dimension of the model
        num_layers: number of layers of the MLP blocks
        is_mlp_residual: whether to use residual connections in the MLP blocks
    """

    def __init__(self, d_model, d, num_layers, is_mlp_residual):
        super().__init__()

        self.w_s = MLP(d_model, d, int((d_model + d) / 2), num_layers, is_mlp_residual)
        self.w_c_cross = MLP(
            d_model, d, int((d_model + d) / 2), num_layers, is_mlp_residual
        )
        self.w_c_global = MLP(d, d, d, num_layers, is_mlp_residual)

        self.output_layer = nn.Linear(d, d_model)

    def forward(self, s, c_cross, c_global):
        """context gating forward function

        Args:

        s: (batch, agents, features) tensor of agent encoded states
        c_cross: (batch, objects, features) tensor of objects encoded states
        c_global: (batch, d) tensor of global context

        Returns:

        s: (batch, agents, features) updated tensor of agent encoded states
        c_global: updated tensor of global context

        """
        s = self.w_s(s)
        c_cross = self.w_c_cross(c_cross)
        c_global = pool(c_cross, -2) * self.w_c_global(c_global)
        # b: batch, a: agents, k: features
        s = torch.einsum("bak,bk->bak", [s, c_global])
        s = self.output_layer(s)
        return s, c_global
