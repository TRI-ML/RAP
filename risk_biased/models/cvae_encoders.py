from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn

from risk_biased.models.cvae_params import CVAEParams
from risk_biased.models.nn_blocks import (
    MCG,
    MAB,
    MHB,
    SequenceEncoderLSTM,
    SequenceEncoderMLP,
    SequenceEncoderMaskedLSTM,
)
from risk_biased.models.latent_distributions import AbstractLatentDistribution


class BaseEncoderNN(nn.Module):
    """Base encoder neural network that defines the common functionality of encoders.
    It should not be used directly but rather extended to define specific encoders.

    Args:
       params: dataclass defining the necessary parameters
       num_steps: length of the input sequence
    """

    def __init__(
        self,
        params: CVAEParams,
        latent_dim: int,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.is_mlp_residual = params.is_mlp_residual
        self.num_hidden_layers = params.num_hidden_layers
        self.num_steps = params.num_steps
        self.num_steps_future = params.num_steps_future
        self.sequence_encoder_type = params.sequence_encoder_type
        self.state_dim = params.state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = params.hidden_dim

        if params.sequence_encoder_type == "MLP":
            self._agent_encoder = SequenceEncoderMLP(
                params.state_dim,
                params.hidden_dim,
                params.num_hidden_layers,
                num_steps,
                params.is_mlp_residual,
            )
        elif params.sequence_encoder_type == "LSTM":
            self._agent_encoder = SequenceEncoderLSTM(
                params.state_dim, params.hidden_dim
            )
        elif params.sequence_encoder_type == "maskedLSTM":
            self._agent_encoder = SequenceEncoderMaskedLSTM(
                params.state_dim, params.hidden_dim
            )

        if params.interaction_type == "Attention" or params.interaction_type == "MAB":
            self._interaction = MAB(
                params.hidden_dim, params.num_attention_heads, params.num_blocks
            )
        elif (
            params.interaction_type == "ContextGating"
            or params.interaction_type == "MCG"
        ):
            self._interaction = MCG(
                params.hidden_dim,
                params.mcg_dim_expansion,
                params.mcg_num_layers,
                params.num_blocks,
                params.is_mlp_residual,
            )
        elif params.interaction_type == "Hybrid" or params.interaction_type == "MHB":
            self._interaction = MHB(
                params.hidden_dim,
                params.num_attention_heads,
                params.mcg_dim_expansion,
                params.mcg_num_layers,
                params.num_blocks,
                params.is_mlp_residual,
            )
        else:
            self._interaction = lambda x, *args, **kwargs: x
        self._output_layer = nn.Linear(params.hidden_dim, self.latent_dim)

    def encode_agents(self, x: torch.Tensor, mask_x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        encoded_absolute: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
        x_ego: Optional[torch.Tensor] = None,
        y_ego: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        risk_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function that encodes input tensors into an output tensor of dimension
        latent_dim.

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            encoded_absolute: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects) tensor of bool mask
            y (optional): (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_y (optional): (batch_size, num_agents, num_steps_future) tensor of bool mask. Defaults to None.
            x_ego: (batch_size, 1, num_steps, state_dim) ego history
            y_ego: (batch_size, 1, num_steps_future, state_dim) ego future
            offset (optional): (batch_size, num_agents, state_dim) offset position from ego.
            risk_level (optional): (batch_size, num_agents) tensor of risk levels desired for future
                trajectories. Defaults to None.

        Returns:
            (batch_size, num_agents, latent_dim) output tensor
        """
        h_agents = self.encode_agents(
            x=x,
            mask_x=mask_x,
            y=y,
            mask_y=mask_y,
            x_ego=x_ego,
            y_ego=y_ego,
            offset=offset,
            risk_level=risk_level,
        )
        mask_agent = mask_x.any(-1)
        h_agents = self._interaction(
            h_agents, mask_agent, encoded_absolute, encoded_map, mask_map
        )

        return self._output_layer(h_agents)


class BiasedEncoderNN(BaseEncoderNN):
    """Biased encoder neural network that encodes past info and auxiliary input
    into a biased distribution over the latent space.

     Args:
        params: dataclass defining the necessary parameters
        num_steps: length of the input sequence
    """

    def __init__(
        self,
        params: CVAEParams,
        latent_dim: int,
        num_steps: int,
    ) -> None:
        super().__init__(params, latent_dim, num_steps)
        self.condition_on_ego_future = params.condition_on_ego_future
        if params.sequence_encoder_type == "MLP":
            self._ego_encoder = SequenceEncoderMLP(
                params.state_dim,
                params.hidden_dim,
                params.num_hidden_layers,
                params.num_steps
                + params.num_steps_future * self.condition_on_ego_future,
                params.is_mlp_residual,
            )
        elif params.sequence_encoder_type == "LSTM":
            self._ego_encoder = SequenceEncoderLSTM(params.state_dim, params.hidden_dim)
        elif params.sequence_encoder_type == "maskedLSTM":
            self._ego_encoder = SequenceEncoderMaskedLSTM(
                params.state_dim, params.hidden_dim
            )

        self._auxiliary_encode = nn.Linear(
            params.hidden_dim + 1 + params.hidden_dim, params.hidden_dim
        )

    def biased_parameters(self, recurse: bool = True):
        """Get the parameters to be optimized when training to bias."""
        yield from self.parameters(recurse)

    def encode_agents(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        *,
        x_ego: torch.Tensor,
        y_ego: torch.Tensor,
        offset: torch.Tensor,
        risk_level: torch.Tensor,
        **kwargs,
    ):
        """Encode agent input and auxiliary input into a feature vector.

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            x_ego: (batch_size, 1, num_steps, state_dim) ego history
            y_ego: (batch_size, 1, num_steps_future, state_dim) ego future
            offset: (batch_size, num_agents, state_dim) offset position from ego.
            risk_level: (batch_size, num_agents) tensor of risk levels desired for future
                trajectories. Defaults to None.
        Returns:
            (batch_size, latent_dim) output tensor
        """

        if self.condition_on_ego_future:
            ego_tensor = torch.cat([x_ego, y_ego], dim=-2)
        else:
            ego_tensor = x_ego

        risk_feature = ((risk_level - 0.5) * 10).exp().unsqueeze(-1)
        mask_ego = torch.ones(
            ego_tensor.shape[0],
            offset.shape[1],
            ego_tensor.shape[2],
            device=ego_tensor.device,
        )
        batch_size, n_agents, dynamic_state_dim = offset.shape
        state_dim = ego_tensor.shape[-1]
        extended_offset = torch.cat(
            (
                offset,
                torch.zeros(
                    batch_size,
                    n_agents,
                    state_dim - dynamic_state_dim,
                    device=offset.device,
                ),
            ),
            dim=-1,
        ).unsqueeze(-2)
        if extended_offset.shape[1] > 1:
            ego_encoded = self._ego_encoder(
                ego_tensor + extended_offset[:, :1] - extended_offset, mask_ego
            )
        else:
            ego_encoded = self._ego_encoder(ego_tensor - extended_offset, mask_ego)
        auxiliary_input = torch.cat((risk_feature, ego_encoded), -1)

        h_agents = self._agent_encoder(x, mask_x)
        h_agents = torch.cat([h_agents, auxiliary_input], dim=-1)
        h_agents = self._auxiliary_encode(h_agents)

        return h_agents


class InferenceEncoderNN(BaseEncoderNN):
    """Inference encoder neural network that encodes past info into the
    inference distribution over the latent space.

    Args:
        params: dataclass defining the necessary parameters
        num_steps: length of the input sequence
    """

    def biaser_parameters(self, recurse: bool = True):
        yield from []

    def encode_agents(self, x: torch.Tensor, mask_x: torch.Tensor, *args, **kwargs):
        h_agents = self._agent_encoder(x, mask_x)
        return h_agents


class FutureEncoderNN(BaseEncoderNN):
    """Future encoder neural network that encodes past and future info into the
    future-conditioned distribution over the latent space.
    The future is not available at test time, this is only used for training.

    Args:
        params: dataclass defining the necessary parameters
        num_steps: length of the input sequence

    """

    def biaser_parameters(self, recurse: bool = True):
        """The future encoder is not optimized when training to bias."""
        yield from []

    def encode_agents(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        *,
        y: torch.Tensor,
        mask_y: torch.Tensor,
        **kwargs,
    ):
        """Encode agent input and future input into a feature vector.
        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of trajectory history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            y: (batch_size, num_agents, num_steps_future, state_dim) future trajectory
            mask_y: (batch_size, num_agents, num_steps_future) tensor of bool mask
        """
        mask_traj = torch.cat([mask_x, mask_y], dim=-1)
        h_agents = self._agent_encoder(torch.cat([x, y], dim=-2), mask_traj)
        return h_agents


class CVAEEncoder(nn.Module):
    """Encoder architecture for conditional variational autoencoder

    Args:
        model: encoder neural network that transforms input tensors to an unsplitted latent output
        latent_distribution_creator: Class that creates a latent distribution class for the latent space.
    """

    def __init__(
        self,
        model: BaseEncoderNN,
        latent_distribution_creator,
    ) -> None:
        super().__init__()
        self._model = model
        self.latent_dim = model.latent_dim
        self._latent_distribution_creator = latent_distribution_creator

    def biased_parameters(self, recurse: bool = True):
        yield from self._model.biased_parameters(recurse)

    def forward(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        encoded_absolute: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
        x_ego: Optional[torch.Tensor] = None,
        y_ego: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        risk_level: Optional[torch.Tensor] = None,
    ) -> AbstractLatentDistribution:
        """Forward function that encodes input tensors into an output tensor of dimension
        latent_dim.

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            encoded_absolute: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects) tensor of bool mask
            y (optional): (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_y (optional): (batch_size, num_agents, num_steps_future) tensor of bool mask. Defaults to None.
            x_ego (optional): (batch_size, 1, num_steps, state_dim) ego history
            y_ego (optional): (batch_size, 1, num_steps_future, state_dim) ego future
            offset (optional): (batch_size, num_agents, state_dim) offset position from ego.
            risk_level (optional): (batch_size, num_agents) tensor of risk levels desired for future
                trajectories. Defaults to None.

        Returns:
            Latent distribution representing the posterior over the latent variables given the input observations.
        """

        latent_output = self._model(
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

        latent_distribution = self._latent_distribution_creator(latent_output)

        return latent_distribution
