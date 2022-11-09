from dataclasses import dataclass

from mmcv import Config


@dataclass
class CVAEParams:
    """
    state_dim: Dimension of the state at each time step.
    map_state_dim: Dimension of the map point features at each position.
    num_steps: Number of time steps in the past trajectory input.
    num_steps_future: Number of time steps in the future trajectory output.
    latent_dim: Dimension of the latent space
    hidden_dim: Dimension of the hidden layers
    num_hidden_layers: Number of layers for each model, (encoder, decoder)
    is_mlp_residual: Set to True to add linear transformation of the input to output of the MLP
    interaction_type: Wether to use MCG, MAB, or MHB to handle interactions
    num_attention_heads: Number of attention heads to use in MHA blocks
    mcg_dim_expansion: Dimension expansion factor for the MCG global interaction space
    mcg_num_layers: Number of layers for the MLP MCG blocks
    num_blocks: Number of interaction blocks to use
    sequence_encoder_type: Type of sequence encoder maskedLSTM, LSTM, or MLP
    sequence_decoder_type: Type of sequence decoder maskedLSTM, LSTM, or MLP
    condition_on_ego_future: Wether to condition the biasing with the ego future or only the ego past
    latent_regularization: Weight of the latent regularization loss
    """

    dt: float
    state_dim: int
    dynamic_state_dim: int
    map_state_dim: int
    max_size_lane: int
    num_steps: int
    num_steps_future: int
    latent_dim: int
    hidden_dim: int
    num_hidden_layers: int
    is_mlp_residual: bool
    interaction_type: int
    num_attention_heads: int
    mcg_dim_expansion: int
    mcg_num_layers: int
    num_blocks: int
    sequence_encoder_type: str
    sequence_decoder_type: str
    condition_on_ego_future: bool
    latent_regularization: float
    risk_assymetry_factor: float
    num_vq: int
    latent_distribution: str

    @staticmethod
    def from_config(cfg: Config):
        return CVAEParams(
            dt=cfg.dt,
            state_dim=cfg.state_dim,
            dynamic_state_dim=cfg.dynamic_state_dim,
            map_state_dim=cfg.map_state_dim,
            max_size_lane=cfg.max_size_lane,
            num_steps=cfg.num_steps,
            num_steps_future=cfg.num_steps_future,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            is_mlp_residual=cfg.is_mlp_residual,
            interaction_type=cfg.interaction_type,
            mcg_dim_expansion=cfg.mcg_dim_expansion,
            mcg_num_layers=cfg.mcg_num_layers,
            num_blocks=cfg.num_blocks,
            num_attention_heads=cfg.num_attention_heads,
            sequence_encoder_type=cfg.sequence_encoder_type,
            sequence_decoder_type=cfg.sequence_decoder_type,
            condition_on_ego_future=cfg.condition_on_ego_future,
            latent_regularization=cfg.latent_regularization,
            risk_assymetry_factor=cfg.risk_assymetry_factor,
            num_vq=cfg.num_vq,
            latent_distribution=cfg.latent_distribution,
        )
