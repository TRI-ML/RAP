from einops import rearrange, repeat
import torch
import torch.nn as nn

from risk_biased.models.cvae_params import CVAEParams
from risk_biased.models.nn_blocks import (
    MCG,
    MAB,
    MHB,
    SequenceDecoderLSTM,
    SequenceDecoderMLP,
    SequenceEncoderLSTM,
    SequenceEncoderMLP,
    SequenceEncoderMaskedLSTM,
)


class DecoderNN(nn.Module):
    """Decoder neural network that decodes input tensors into a single output tensor.
    It contains an interaction layer that (re-)compute the interactions between the agents in the scene.
    This implies that a given latent sample for one agent will be affecting the predictions of the othe agents too.

    Args:
        params: dataclass defining the necessary parameters

    """

    def __init__(
        self,
        params: CVAEParams,
    ) -> None:
        super().__init__()
        self.dt = params.dt
        self.state_dim = params.state_dim
        self.dynamic_state_dim = params.dynamic_state_dim
        self.hidden_dim = params.hidden_dim
        self.num_steps_future = params.num_steps_future
        self.latent_dim = params.latent_dim

        if params.sequence_encoder_type == "MLP":
            self._agent_encoder_past = SequenceEncoderMLP(
                params.state_dim,
                params.hidden_dim,
                params.num_hidden_layers,
                params.num_steps,
                params.is_mlp_residual,
            )
        elif params.sequence_encoder_type == "LSTM":
            self._agent_encoder_past = SequenceEncoderLSTM(
                params.state_dim, params.hidden_dim
            )
        elif params.sequence_encoder_type == "maskedLSTM":
            self._agent_encoder_past = SequenceEncoderMaskedLSTM(
                params.state_dim, params.hidden_dim
            )
        else:
            raise RuntimeError(
                f"Got sequence encoder type {params.sequence_decoder_type} but only knows one of: 'MLP', 'LSTM', 'maskedLSTM' "
            )

        self._combine_z_past = nn.Linear(
            params.hidden_dim + params.latent_dim, params.hidden_dim
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

        if params.sequence_decoder_type == "MLP":
            self._decoder = SequenceDecoderMLP(
                params.hidden_dim,
                params.num_hidden_layers,
                params.num_steps_future,
                params.is_mlp_residual,
            )
        elif params.sequence_decoder_type == "LSTM":
            self._decoder = SequenceDecoderLSTM(params.hidden_dim)
        elif params.sequence_decoder_type == "maskedLSTM":
            self._decoder = SequenceDecoderLSTM(params.hidden_dim)
        else:
            raise RuntimeError(
                f"Got sequence decoder type {params.sequence_decoder_type} but only knows one of: 'MLP', 'LSTM', 'maskedLSTM' "
            )

    def forward(
        self,
        z_samples: torch.Tensor,
        mask_z: torch.Tensor,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        encoded_absolute: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function that decodes input tensors into an output tensor of size
        (batch_size, num_agents, (n_samples), num_steps_future, state_dim)

        Args:
            z_samples: (batch_size, num_agents, (n_samples), latent_dim) tensor of history
            mask_z: (batch_size, num_agents) tensor of bool mask
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history for all agents
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            encoded_absolute: (batch_size, num_agents,  feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects) tensor of bool mask

        Returns:
            (batch_size, num_agents, (n_samples), num_steps_future, state_dim) output tensor
        """

        encoded_x = self._agent_encoder_past(x, mask_x)
        squeeze_output_sample_dim = False
        if z_samples.ndim == 3:
            batch_size, num_agents, latent_dim = z_samples.shape
            num_samples = 1
            z_samples = rearrange(z_samples, "b a l -> b a () l")
            squeeze_output_sample_dim = True
        else:
            batch_size, num_agents, num_samples, latent_dim = z_samples.shape
            mask_z = repeat(mask_z, "b a -> (b s) a", s=num_samples)
            mask_map = repeat(mask_map, "b o -> (b s) o", s=num_samples)
            encoded_x = repeat(encoded_x, "b a l -> (b s) a l", s=num_samples)
            encoded_absolute = repeat(
                encoded_absolute, "b a l -> (b s) a l", s=num_samples
            )
            encoded_map = repeat(encoded_map, "b o l -> (b s) o l", s=num_samples)

        z_samples = rearrange(z_samples, "b a s l -> (b s) a l")

        h = self._combine_z_past(torch.cat([z_samples, encoded_x], dim=-1))

        h = self._interaction(h, mask_z, encoded_absolute, encoded_map, mask_map)

        h = self._decoder(h, self.num_steps_future)

        if not squeeze_output_sample_dim:
            h = rearrange(h, "(b s) a t l -> b a s t l", b=batch_size, s=num_samples)

        return h


class CVAEAccelerationDecoder(nn.Module):
    """Decoder architecture for conditional variational autoencoder

    Args:
        model: decoder neural network that transforms input tensors to an output sequence
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self._model = model
        self._output_layer = nn.Linear(model.hidden_dim, 2)

    def forward(
        self,
        z_samples: torch.Tensor,
        mask_z: torch.Tensor,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        encoded_absolute: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function that decodes input tensors into an output tensor of size
        (batch_size, num_agents, (n_samples), num_steps_future, state_dim=5)
        It first predicts accelerations that are doubly integrated to produce the output
        state sequence with positions angles and velocities (x, y, theta, vx, vy) or (x, y, vx, vy) or (x, y)

        Args:
            z_samples: (batch_size, num_agents, (n_samples), latent_dim) tensor of history
            mask_z: (batch_size, num_agents) tensor of bool mask
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history for all agents
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            encoded_absolute: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects) tensor of bool mask

        Returns:
            (batch_size, num_agents, (n_samples), num_steps_future, state_dim) output tensor. Sample dimension
            does not exist if z_samples is a 2D tensor.
        """

        h = self._model(
            z_samples, mask_z, x, mask_x, encoded_absolute, encoded_map, mask_map
        )
        h = self._output_layer(h)

        dt = self._model.dt
        initial_position = x[..., -1:, :2].clone()
        # If shape is 5 it should be (x, y, angle, vx, vy)
        if offset.shape[-1] == 5:
            initial_velocity = offset[..., 3:5].clone().unsqueeze(-2)
        # else if shape is 4 it should be (x, y, vx, vy)
        elif offset.shape[-1] == 4:
            initial_velocity = offset[..., 2:4].clone().unsqueeze(-2)
        elif x.shape[-1] == 5:
            initial_velocity = x[..., -1:, 3:5].clone()
        elif x.shape[-1] == 4:
            initial_velocity = x[..., -1:, 2:4].clone()
        else:
            initial_velocity = (x[..., -1:, :] - x[..., -2:-1, :]) / dt

        output = torch.zeros(
            (*h.shape[:-1], self._model.dynamic_state_dim), device=h.device
        )
        # There might be a sample dimension in the output tensor, then adapt the shape of initial position and velocity
        if output.ndim == 5:
            initial_position = initial_position.unsqueeze(-3)
            initial_velocity = initial_velocity.unsqueeze(-3)

        if self._model.dynamic_state_dim == 5:
            output[..., 3:5] = h.cumsum(-2) * dt
            output[..., :2] = (output[..., 3:5].clone() + initial_velocity).cumsum(
                -2
            ) * dt + initial_position
            output[..., 2] = torch.atan2(output[..., 4].clone(), output[..., 3].clone())
        elif self._model.dynamic_state_dim == 4:
            output[..., 2:4] = h.cumsum(-2) * dt
            output[..., :2] = (output[..., 2:4].clone() + initial_velocity).cumsum(
                -2
            ) * dt + initial_position
        else:
            velocity = h.cumsum(-2) * dt
            output = (velocity.clone() + initial_velocity).cumsum(
                -2
            ) * dt + initial_position
        return output


class CVAEParametrizedDecoder(nn.Module):
    """Decoder architecture for conditional variational autoencoder

    Args:
        model: decoder neural network that transforms input tensors to an output sequence
    """

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self._model = model
        self._order = 3
        self._output_layer = nn.Linear(
            model.hidden_dim * model.num_steps_future,
            2 * self._order + model.num_steps_future,
        )

    def polynomial(self, x: torch.Tensor, params: torch.Tensor):
        """Polynomial function that takes a tensor of shape (batch_size, num_agents, (n_samples), num_steps_future) and
        a parameter tensor of shape (batch_size, num_agents, (n_samples), self._order*2) and returns a tensor of shape (batch_size, num_agents, (n_samples), num_steps_future)
        """
        h = x.clone()
        squeeze = False
        if h.ndim == 3:
            h = h.unsqueeze(2)
            params = params.unsqueeze(2)
            squeeze = True
        h = repeat(
            h,
            "batch agents samples sequence -> batch agents samples sequence two order",
            order=self._order,
            two=2,
        ).cumprod(-1)
        h = h * params.view(*params.shape[:-1], 1, 2, self._order)
        h = h.sum(-1)
        if squeeze:
            h = h.squeeze(2)
        return h

    def dpolynomial(self, x: torch.Tensor, params: torch.Tensor):
        """Derivative of the polynomial function that takes a tensor of shape (batch_size, num_agents, (n_samples), num_steps_future) and
        a parameter tensor of shape (batch_size, num_agents, (n_samples), self._order*2) and returns a tensor of shape (batch_size, num_agents, (n_samples), num_steps_future)
        """
        h = x.clone()
        squeeze = False
        if h.ndim == 3:
            h = h.unsqueeze(2)
            params = params.unsqueeze(2)
            squeeze = True
        h = repeat(
            h,
            "batch agents samples sequence -> batch agents samples sequence two order",
            order=self._order - 1,
            two=2,
        )
        h = torch.cat((torch.ones_like(h[..., :1]), h.cumprod(-1)), -1)
        h = h * params.view(*params.shape[:-1], 1, 2, self._order)
        h = h * torch.arange(self._order).view(*([1] * params.ndim), -1).to(x.device)
        h = h.sum(-1)
        if squeeze:
            h = h.squeeze(2)
        return h

    def forward(
        self,
        z_samples: torch.Tensor,
        mask_z: torch.Tensor,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        encoded_absolute: torch.Tensor,
        encoded_map: torch.Tensor,
        mask_map: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function that decodes input tensors into an output tensor of size
        (batch_size, num_agents, (n_samples), num_steps_future, state_dim=5)
        It first predicts accelerations that are doubly integrated to produce the output
        state sequence with positions angles and velocities (x, y, theta, vx, vy) or (x, y, vx, vy) or (x, y)

        Args:
            z_samples: (batch_size, num_agents, (n_samples), latent_dim) tensor of history
            mask_z: (batch_size, num_agents) tensor of bool mask
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history for all agents
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            encoded_absolute: (batch_size, num_agents, feature_size) tensor of the encoded absolute agent positions
            encoded_map: (batch_size, num_objects, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects) tensor of bool mask

        Returns:
            (batch_size, num_agents, (n_samples), num_steps_future, state_dim) output tensor. Sample dimension
            does not exist if z_samples is a 2D tensor.
        """

        squeeze_output_sample_dim = z_samples.ndim == 3
        batch_size = z_samples.shape[0]

        h = self._model(
            z_samples, mask_z, x, mask_x, encoded_absolute, encoded_map, mask_map
        )
        if squeeze_output_sample_dim:
            h = rearrange(
                h, "batch agents sequence features -> batch agents (sequence features)"
            )
        else:
            h = rearrange(
                h,
                "(batch samples) agents sequence features -> batch agents samples (sequence features)",
                batch=batch_size,
            )
        h = self._output_layer(h)

        output = torch.zeros(
            (
                *h.shape[:-1],
                self._model.num_steps_future,
                self._model.dynamic_state_dim,
            ),
            device=h.device,
        )
        params = h[..., : 2 * self._order]
        dldt = torch.relu(h[..., 2 * self._order :])
        distance = dldt.cumsum(-2)
        output[..., :2] = self.polynomial(distance, params)
        if self._model.dynamic_state_dim == 5:
            output[..., 3:5] = dldt * self.dpolynomial(distance, params)
            output[..., 2] = torch.atan2(output[..., 4].clone(), output[..., 3].clone())
        elif self._model.dynamic_state_dim == 4:
            output[..., 2:4] = dldt * self.dpolynomial(distance, params)

        return output
