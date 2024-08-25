from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch
import torch.nn as nn

from risk_biased.models.multi_head_attention import MultiHeadAttention
from risk_biased.models.context_gating import ContextGating
from risk_biased.models.mlp import MLP


def identity(x, *args, **kwargs):
    return x

class SequenceEncoderMaskedLSTM(nn.Module):
    """MLP followed with a masked LSTM implementation with one layer.

    Args:
        input_dim : dimension of the input variable
        h_dim : dimension of a hidden layer of MLP
    """

    def __init__(self, input_dim: int, h_dim: int) -> None:
        super().__init__()
        self._group_objects = Rearrange("b o ... -> (b o) ...")
        self._embed = nn.Linear(in_features=input_dim, out_features=h_dim)
        self._lstm = nn.LSTMCell(
            input_size=h_dim, hidden_size=h_dim
        )  # expects(batch,seq,features)
        self.h0 = nn.parameter.Parameter(torch.zeros(1, h_dim))
        self.c0 = nn.parameter.Parameter(torch.zeros(1, h_dim))

    def forward(self, input: torch.Tensor, mask_input: torch.Tensor) -> torch.Tensor:
        """Forward function for MapEncoder

        Args:
            input (torch.Tensor): (batch_size, num_objects, seq_len, input_dim) tensor
            mask_input (torch.Tensor): (batch_size, num_objects, seq_len) bool tensor (True if data is good False if data is missing)

        Returns:
            torch.Tensor: (batch_size, num_objects, output_dim) tensor
        """

        batch_size, num_objects, seq_len, _ = input.shape
        split_objects = Rearrange("(b o) f -> b o f", b=batch_size, o=num_objects)

        input = self._group_objects(input)
        mask_input = self._group_objects(mask_input)
        embedded_input = self._embed(input)

        # One to many encoding of the input sequence with masking for missing points
        mask_input = mask_input.float()
        h = mask_input[:, 0, None] * embedded_input[:, 0, :] + (
            1 - mask_input[:, 0, None]
        ) * repeat(self.h0, "b f -> (size b) f", size=batch_size * num_objects)
        c = repeat(self.c0, "b f -> (size b) f", size=batch_size * num_objects)
        for i in range(seq_len):
            new_input = (
                mask_input[:, i, None] * embedded_input[:, i, :]
                + (1 - mask_input[:, i, None]) * h
            )
            h, c = self._lstm(new_input, (h, c))
        return split_objects(h)


class SequenceEncoderLSTM(nn.Module):
    """MLP followed with an LSTM with one layer.

    Args:
        input_dim : dimension of the input variable
        h_dim : dimension of a hidden layer of MLP
    """

    def __init__(self, input_dim: int, h_dim: int) -> None:
        super().__init__()
        self._group_objects = Rearrange("b o ... -> (b o) ...")
        self._embed = nn.Linear(in_features=input_dim, out_features=h_dim)
        self._lstm = nn.LSTM(
            input_size=h_dim,
            hidden_size=h_dim,
            batch_first=True,
        )  # expects(batch,seq,features)
        self.h0 = nn.parameter.Parameter(torch.zeros(1, h_dim))
        self.c0 = nn.parameter.Parameter(torch.zeros(1, h_dim))

    def forward(self, input: torch.Tensor, mask_input: torch.Tensor) -> torch.Tensor:
        """Forward function for MapEncoder

        Args:
            input (torch.Tensor): (batch_size, num_objects, seq_len, input_dim) tensor
            mask_input (torch.Tensor): (batch_size, num_objects, seq_len) bool tensor (True if data is good False if data is missing)

        Returns:
            torch.Tensor: (batch_size, num_objects, output_dim) tensor
        """

        batch_size, num_objects, seq_len, _ = input.shape
        split_objects = Rearrange("(b o) f -> b o f", b=batch_size, o=num_objects)

        input = self._group_objects(input)
        mask_input = self._group_objects(mask_input)
        embedded_input = self._embed(input)

        # One to many encoding of the input sequence with masking for missing points
        mask_input = mask_input.float()
        h = (
            mask_input[:, 0, None] * embedded_input[:, 0, :]
            + (1 - mask_input[:, 0, None])
            * repeat(
                self.h0, "one f -> one size f", size=batch_size * num_objects
            ).contiguous()
        )
        c = repeat(
            self.c0, "one f -> one size f", size=batch_size * num_objects
        ).contiguous()
        _, (h, _) = self._lstm(embedded_input, (h, c))
        # for i in range(seq_len):
        #     new_input = (
        #         mask_input[:, i, None] * embedded_input[:, i, :]
        #         + (1 - mask_input[:, i, None]) * h
        #     )
        #     h, c = self._lstm(new_input, (h, c))
        return split_objects(h.squeeze(0))


class SequenceEncoderMLP(nn.Module):
    """MLP implementation.

    Args:
        input_dim : dimension of the input variable
        h_dim : dimension of a hidden layer of MLP
        num_layers: number of layers to use in the MLP
        sequence_length: dimension of the input sequence
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        num_layers: int,
        sequence_length: int,
        is_mlp_residual: bool,
    ) -> None:
        super().__init__()
        self._mlp = MLP(
            input_dim * sequence_length, h_dim, h_dim, num_layers, is_mlp_residual
        )

    def forward(self, input: torch.Tensor, mask_input: torch.Tensor) -> torch.Tensor:
        """Forward function for MapEncoder

        Args:
            input (torch.Tensor): (batch_size, num_objects, seq_len, input_dim) tensor
            mask_input (torch.Tensor): (batch_size, num_objects, seq_len) bool tensor (True if data is good False if data is missing)

        Returns:
            torch.Tensor: (batch_size, num_objects, output_dim) tensor
        """

        batch_size, num_objects, _, _ = input.shape
        input = input * mask_input.unsqueeze(-1)
        h = rearrange(input, "b o s f -> (b o) (s f)")
        mask_input = rearrange(mask_input, "b o s -> (b o) s")
        if h.shape[-1] == 0:
            h = h.view(batch_size, 0, h.shape[0])
        else:
            h = self._mlp(h)
            h = rearrange(h, "(b o) f -> b o f", b=batch_size, o=num_objects)

        return h


class SequenceDecoderLSTM(nn.Module):
    """A one to many LSTM implementation with one layer.

    Args:
        h_dim : dimension of a hidden layer
    """

    def __init__(self, h_dim: int) -> None:
        super().__init__()
        self._group_objects = Rearrange("b o f -> (b o) f")
        self._lstm = nn.LSTM(input_size=h_dim, hidden_size=h_dim)
        self._out_layer = nn.Linear(in_features=h_dim, out_features=h_dim)
        self.c0 = nn.parameter.Parameter(torch.zeros(1, h_dim))

    def forward(self, input: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Forward function for MapEncoder

        Args:
            input (torch.Tensor): (batch_size, num_objects, input_dim) tensor
            sequence_length: output sequence length to create
        Returns:
            torch.Tensor: (batch_size, num_objects, output_dim) tensor
        """

        batch_size, num_objects, _ = input.shape

        h = repeat(input, "b o f -> one (b o) f", one=1).contiguous()
        c = repeat(
            self.c0, "one f -> one size f", size=batch_size * num_objects
        ).contiguous()
        seq_h = repeat(h, "one b f -> (one t) b f", t=sequence_length).contiguous()
        h, (_, _) = self._lstm(seq_h, (h, c))
        h = rearrange(h, "t (b o) f -> b o t f", b=batch_size, o=num_objects)
        return self._out_layer(h)


class SequenceDecoderMLP(nn.Module):
    """A one to many MLP implementation.

    Args:
        h_dim : dimension of a hidden layer
        num_layers: number of layers to use in the MLP
        sequence_length: output sequence length to return
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self, h_dim: int, num_layers: int, sequence_length: int, is_mlp_residual: bool
    ) -> None:
        super().__init__()
        self._mlp = MLP(
            h_dim, h_dim * sequence_length, h_dim, num_layers, is_mlp_residual
        )

    def forward(self, input: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Forward function for MapEncoder

        Args:
            input (torch.Tensor): (batch_size, num_objects, input_dim) tensor
            sequence_length: output sequence length to create
        Returns:
            torch.Tensor: (batch_size, num_objects, output_dim) tensor
        """

        batch_size, num_objects, _ = input.shape

        h = rearrange(input, "b o f -> (b o) f")
        h = self._mlp(h)
        h = rearrange(
            h, "(b o) (s f) -> b o s f", b=batch_size, o=num_objects, s=sequence_length
        )
        return h


class AttentionBlock(nn.Module):
    """Block performing agent-map cross attention->ReLU(linear)->+residual->layer_norm->agent-agent attention->ReLU(linear)->+residual->layer_norm
    Args:
        hidden_dim: feature dimension
        num_attention_heads: number of attention heads to use
    """

    def __init__(self, hidden_dim: int, num_attention_heads: int):
        super().__init__()
        self._num_attention_heads = num_attention_heads
        self._agent_map_attention = MultiHeadAttention(
            hidden_dim, num_attention_heads, hidden_dim, hidden_dim
        )
        self._lin1 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm1 = nn.LayerNorm(hidden_dim)
        self._agent_agent_attention = MultiHeadAttention(
            hidden_dim, num_attention_heads, hidden_dim, hidden_dim
        )
        self._lin2 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm2 = nn.LayerNorm(hidden_dim)
        self._activation = nn.ReLU()

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning only the output (no attention matrix)
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
        """

        # Check if map_info is available. If not, don't compute cross-attention with it
        if mask_map.any():
            mask_agent_map = torch.einsum("ba,bo->bao", mask_agents, mask_map)
            h, _ = self._agent_map_attention(
                encoded_agents + encoded_absolute_agents,
                encoded_map,
                encoded_map,
                mask=mask_agent_map,
            )
            h = torch.masked_fill(h, torch.logical_not(mask_agents.unsqueeze(-1)), 0)
            h = torch.sigmoid(self._lin1(h))
            h = self._layer_norm1(encoded_agents + h)
        else:
            h = self._layer_norm1(encoded_agents)

        h_res = h.clone()
        agent_agent_mask = torch.einsum("ba,be->bae", mask_agents, mask_agents)
        h = h + encoded_absolute_agents
        h, _ = self._agent_agent_attention(h, h, h, mask=agent_agent_mask)
        h = torch.masked_fill(h, torch.logical_not(mask_agents.unsqueeze(-1)), 0)
        h = self._activation(self._lin2(h))
        h = self._layer_norm2(h_res + h)
        return h


class CG_block(nn.Module):
    """Block performing context gating agent-map
    Args:
        hidden_dim: feature dimension
        dim_expansion: multiplicative factor on the hidden dimension for the global context representation
        num_layers: number of layers to use in the MLP for context encoding
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        dim_expansion: int,
        num_layers: int,
        is_mlp_residual: bool,
    ):
        super().__init__()
        self._agent_map = ContextGating(
            hidden_dim,
            hidden_dim * dim_expansion,
            num_layers=num_layers,
            is_mlp_residual=is_mlp_residual,
        )
        self._lin1 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm1 = nn.LayerNorm(hidden_dim)
        self._agent_agent = ContextGating(
            hidden_dim, hidden_dim * dim_expansion, num_layers, is_mlp_residual
        )
        self._lin2 = nn.Linear(hidden_dim, hidden_dim)
        self._activation = nn.ReLU()

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning the output and global context
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
            global_context: (batch_size, dim_context) tensor representing the global context
        """

        # Check if map_info is available. If not, don't compute cross-interaction with it
        if mask_map.any():
            s, global_context = self._agent_map(
                encoded_agents + encoded_absolute_agents, encoded_map, global_context
            )
            s = s * mask_agents.unsqueeze(-1)
            s = self._activation(self._lin1(s))
            s = self._layer_norm1(encoded_agents + s)
        else:
            s = self._layer_norm1(encoded_agents)

        s = s + encoded_absolute_agents
        s, global_context = self._agent_agent(s, s, global_context)
        s = s * mask_agents.unsqueeze(-1)
        s = self._lin2(s)
        return s, global_context


class HybridBlock(nn.Module):
    """Block performing agent-map cross context_gating->ReLU(linear)->+residual->layer_norm->agent-agent attention->ReLU(linear)->+residual->layer_norm
    Args:
        hidden_dim: feature dimension
        num_attention_heads: number of attention heads to use
        dim_expansion: multiplicative factor on the hidden dimension for the global context representation
        num_layers: number of layers to use in the MLP for context encoding
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        dim_expansion: int,
        num_layers: int,
        is_mlp_residual: bool,
    ):
        super().__init__()
        self._num_attention_heads = num_attention_heads
        self._agent_map_cg = ContextGating(
            hidden_dim,
            hidden_dim * dim_expansion,
            num_layers=num_layers,
            is_mlp_residual=is_mlp_residual,
        )
        self._lin1 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm1 = nn.LayerNorm(hidden_dim)
        self._agent_agent_attention = MultiHeadAttention(
            hidden_dim, num_attention_heads, hidden_dim, hidden_dim
        )
        self._lin2 = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm2 = nn.LayerNorm(hidden_dim)
        self._activation = nn.ReLU()

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning the output and the context (no attention matrix)
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
            global_context: (batch_size, dim_context) tensor representing the global context
        """

        # Check if map_info is available. If not, don't compute cross-context gating with it
        if mask_map.any():
            # mask_agent_map = torch.logical_not(
            #     torch.einsum("ba,bo->bao", mask_agents, mask_map)
            # )
            h, global_context = self._agent_map_cg(
                encoded_agents + encoded_absolute_agents, encoded_map, global_context
            )
            h = torch.masked_fill(h, torch.logical_not(mask_agents.unsqueeze(-1)), 0)
            h = self._activation(self._lin1(h))
            h = self._layer_norm1(encoded_agents + h)
        else:
            h = self._layer_norm1(encoded_agents)

        h_res = h.clone()
        agent_agent_mask = torch.einsum("ba,be->bae", mask_agents, mask_agents)
        h = h + encoded_absolute_agents
        h, _ = self._agent_agent_attention(h, h, h, mask=agent_agent_mask)
        h = torch.masked_fill(h, torch.logical_not(mask_agents.unsqueeze(-1)), 0)
        h = self._activation(self._lin2(h))
        h = self._layer_norm2(h_res + h)
        return h, global_context


class MCG(nn.Module):
    """Multiple context encoding blocks
    Args:
        hidden_dim: feature dimension
        dim_expansion: multiplicative factor on the hidden dimension for the global context representation
        num_layers: number of layers to use in the MLP for context encoding
        num_blocks: number of successive context encoding blocks to use in the module
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        dim_expansion: int,
        num_layers: int,
        num_blocks: int,
        is_mlp_residual: bool,
    ):
        super().__init__()
        self.initial_global_context = nn.parameter.Parameter(
            torch.ones(1, hidden_dim * dim_expansion)
        )
        list_cg = []
        for i in range(num_blocks):
            list_cg.append(
                CG_block(hidden_dim, dim_expansion, num_layers, is_mlp_residual)
            )
        self.mcg = nn.ModuleList(list_cg)

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning only the output (no context)
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
        """
        s = encoded_agents
        c = self.initial_global_context
        sum_s = s
        sum_c = c
        for i, cg in enumerate(self.mcg):
            s_new, c_new = cg(
                s, mask_agents, encoded_absolute_agents, encoded_map, mask_map, c
            )
            sum_s = sum_s + s_new
            sum_c = sum_c + c_new
            s = (sum_s / (i + 2)).clone()
            c = (sum_c / (i + 2)).clone()
        return s


class MAB(nn.Module):
    """Multiple Attention Blocks
    Args:
        hidden_dim: feature dimension
        num_attention_heads: number of attention heads to use
        num_blocks: number of successive blocks to use in the module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        num_blocks: int,
    ):
        super().__init__()
        list_attention = []
        for i in range(num_blocks):
            list_attention.append(AttentionBlock(hidden_dim, num_attention_heads))
        self.attention_blocks = nn.ModuleList(list_attention)

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning only the output (no attention matrix)
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
        """
        h = encoded_agents
        sum_h = h
        for i, attention in enumerate(self.attention_blocks):
            h_new = attention(
                h, mask_agents, encoded_absolute_agents, encoded_map, mask_map
            )
            sum_h = sum_h + h_new
            h = (sum_h / (i + 2)).clone()
        return h


class MHB(nn.Module):
    """Multiple Hybrid Blocks
    Args:
        hidden_dim: feature dimension
        num_attention_heads: number of attention heads to use
        dim_expansion: multiplicative factor on the hidden dimension for the global context representation
        num_layers: number of layers to use in the MLP for context encoding
        num_blocks: number of successive blocks to use in the module.
        is_mlp_residual: set to True to add a linear transformation of the input to the output of the MLP
    """

    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        dim_expansion: int,
        num_layers: int,
        num_blocks: int,
        is_mlp_residual: bool,
    ):
        super().__init__()
        self.initial_global_context = nn.parameter.Parameter(
            torch.ones(1, hidden_dim * dim_expansion)
        )
        list_hb = []
        for i in range(num_blocks):
            list_hb.append(
                HybridBlock(
                    hidden_dim,
                    num_attention_heads,
                    dim_expansion,
                    num_layers,
                    is_mlp_residual,
                )
            )
        self.hybrid_blocks = nn.ModuleList(list_hb)

    def forward(
        self,
        encoded_agents: torch.Tensor,
        mask_agents: torch.Tensor,
        encoded_absolute_agents: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the block, returning only the output (no attention matrix nor context)
        Args:
            encoded_agents: (batch_size, num_agents, feature_size) tensor of the encoded agent tracks
            mask_agents: (batch_size, num_agents) tensor True if agent track is good False if it is just padding
            encoded_absolute_agents: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, feature_size) tensor of the encoded map object features
            mask_map: (batch_size, num_objects) tensor True if object is good False if it is just padding
        """
        sum_h = encoded_agents
        sum_c = self.initial_global_context
        h = encoded_agents
        c = self.initial_global_context
        for i, hb in enumerate(self.hybrid_blocks):
            h_new, c_new = hb(
                h, mask_agents, encoded_absolute_agents, encoded_map, mask_map, c
            )
            sum_h = sum_h + h_new
            sum_c = sum_c + c_new
            h = (sum_h / (i + 2)).clone()
            c = (sum_c / (i + 2)).clone()
        return h
