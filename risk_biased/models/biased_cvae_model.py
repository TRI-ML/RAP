from warnings import warn
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from risk_biased.models.map_encoder import MapEncoderNN
from risk_biased.models.mlp import MLP
from risk_biased.models.cvae_params import CVAEParams
from risk_biased.models.cvae_encoders import (
    CVAEEncoder,
    BiasedEncoderNN,
    FutureEncoderNN,
    InferenceEncoderNN,
)
from risk_biased.models.cvae_decoder import (
    CVAEAccelerationDecoder,
    CVAEParametrizedDecoder,
    DecoderNN,
)
from risk_biased.utils.cost import BaseCostTorch, get_cost
from risk_biased.utils.loss import (
    reconstruction_loss,
    risk_loss_function,
)
from risk_biased.models.latent_distributions import (
    GaussianLatentDistribution,
    QuantizedDistributionCreator,
    AbstractLatentDistribution,
)
from risk_biased.utils.metrics import FDE, minFDE
from risk_biased.utils.risk import AbstractMonteCarloRiskEstimator


class InferenceBiasedCVAE(nn.Module):
    """CVAE with a biased encoder module for risk-biased trajectory forecasting.

    Args:
        absolute_encoder: encoder model for the absolute positions of the agents
        map_encoder: encoder model for map objects
        biased_encoder: biased encoder that uses past and auxiliary input,
        inference_encoder: inference encoder that uses only past,
        decoder: CVAE decoder model
        prior_distribution: prior distribution for the latent space.
    """

    def __init__(
        self,
        absolute_encoder: MLP,
        map_encoder: MapEncoderNN,
        biased_encoder: CVAEEncoder,
        inference_encoder: CVAEEncoder,
        decoder: CVAEAccelerationDecoder,
        prior_distribution: AbstractLatentDistribution,
    ) -> None:
        super().__init__()
        self.biased_encoder = biased_encoder
        self.inference_encoder = inference_encoder
        self.decoder = decoder
        self.map_encoder = map_encoder
        self.absolute_encoder = absolute_encoder
        self.prior_distribution = prior_distribution

    def cvae_parameters(self, recurse: bool = True):
        """Define an iterator over all the parameters related to the cvae."""
        yield from self.absolute_encoder.parameters(recurse=recurse)
        yield from self.map_encoder.parameters(recurse=recurse)
        yield from self.inference_encoder.parameters(recurse=recurse)
        yield from self.decoder.parameters(recurse=recurse)

    def biased_parameters(self, recurse: bool = True):
        """Define an iterator over only the parameters related to the biaser."""
        yield from self.biased_encoder.biased_parameters(recurse=recurse)

    def forward(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        offset: torch.Tensor,
        *,
        x_ego: Optional[torch.Tensor] = None,
        y_ego: Optional[torch.Tensor] = None,
        risk_level: Optional[torch.Tensor] = None,
        n_samples: int = 0,
    ) -> Tuple[torch.Tensor, AbstractLatentDistribution]:
        """Forward function that outputs a noisy reconstruction of y and parameters of latent
        posterior distribution

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor of bool mask
            offset : (batch_size, num_agents, state_dim) offset position from ego. Defaults to None.
            x_ego: (batch_size, 1, num_steps, state_dim) ego history
            y_ego: (batch_size, 1, num_steps_future, state_dim) ego future
            risk_level (optional): (batch_size, num_agents) tensor of risk levels desired for future
                trajectories. Defaults to None.
            n_samples (optional): number of samples to predict, (if 0 one sample with no extra
                dimension). Defaults to 0.

        Returns:
            noisy reconstruction y of size (batch_size, num_agents, num_steps_future, state_dim), as well as
            weights of the samples and the latent distribution.
            No bias is applied to encoder without offset or risk.
        """

        encoded_map = self.map_encoder(map, mask_map)
        mask_map = mask_map.any(-1)
        encoded_absolute = self.absolute_encoder(offset)

        if risk_level is not None:
            biased_latent_distribution = self.biased_encoder(
                x,
                mask_x,
                encoded_absolute,
                encoded_map,
                mask_map,
                x_ego=x_ego,
                y_ego=y_ego,
                offset=offset,
                risk_level=risk_level,
            )
            inference_latent_distribution = self.inference_encoder(
                x,
                mask_x,
                encoded_absolute,
                encoded_map,
                mask_map,
            )
            latent_distribution = inference_latent_distribution.average(
                biased_latent_distribution, risk_level.unsqueeze(-1)
            )
        else:
            latent_distribution = self.inference_encoder(
                x,
                mask_x,
                encoded_absolute,
                encoded_map,
                mask_map,
            )
        z_sample, weights = latent_distribution.sample(n_samples=n_samples)

        mask_z = mask_x.any(-1)
        y_sample = self.decoder(
            z_sample, mask_z, x, mask_x, encoded_absolute, encoded_map, mask_map, offset
        )

        return y_sample, weights, latent_distribution

    def decode(
        self,
        z_samples: torch.Tensor,
        mask_z: torch.Tensor,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        offset: torch.Tensor,
    ):
        """Returns predicted y values conditionned on z_samples and the other observations.

        Args:
            z_samples: (batch_size, num_agents, (n_samples), latent_dim) tensor of latent samples
            mask_z: (batch_size, num_agents) bool mask
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor True where map features are good False where it is padding
            offset : (batch_size, num_agents, state_dim) offset position from ego.
        """
        encoded_map = self.map_encoder(map, mask_map)
        mask_map = mask_map.any(-1)
        encoded_absolute = self.absolute_encoder(offset)

        return self.decoder(
            z_samples=z_samples,
            mask_z=mask_z,
            x=x,
            mask_x=mask_x,
            encoded_absolute=encoded_absolute,
            encoded_map=encoded_map,
            mask_map=mask_map,
            offset=offset,
        )


class TrainingBiasedCVAE(InferenceBiasedCVAE):

    """CVAE with a biased encoder module for risk-biased trajectory forecasting.
    This module is as a non-sampling-based version of BiasedLatentCVAE.

    Args:
        absolute_encoder: encoder model for the absolute positions of the agents
        map_encoder: encoder model for map objects
        biased_encoder: biased encoder that uses past and auxiliary input,
        inference_encoder: inference encoder that uses only past,
        decoder: CVAE decoder model
        future_encoder: training encoder that uses past and future,
        cost_function: cost function used to compute the risk objective
        risk_estimator: risk estimator used to compute the risk objective
        prior_distribution: prior distribution for the latent space.
        training_mode (optional): set to "cvae" to train the unbiased model, set to "bias" to train
            the biased encoder. Defaults to "cvae".
        latent_regularization (optional): regularization term for the latent space. Defaults to 0.
        risk_assymetry_factor (optional): risk asymmetry factor used to compute the risk objective avoiding underestimations.
    """

    def __init__(
        self,
        absolute_encoder: MLP,
        map_encoder: MapEncoderNN,
        biased_encoder: CVAEEncoder,
        inference_encoder: CVAEEncoder,
        decoder: CVAEAccelerationDecoder,
        future_encoder: CVAEEncoder,
        cost_function: BaseCostTorch,
        risk_estimator: AbstractMonteCarloRiskEstimator,
        prior_distribution: AbstractLatentDistribution,
        training_mode: str = "cvae",
        latent_regularization: float = 0.0,
        risk_assymetry_factor: float = 100.0,
    ) -> None:
        super().__init__(
            absolute_encoder,
            map_encoder,
            biased_encoder,
            inference_encoder,
            decoder,
            prior_distribution,
        )
        self.future_encoder = future_encoder
        self._cost = cost_function
        self._risk = risk_estimator
        self.set_training_mode(training_mode)
        self.regularization_factor = latent_regularization
        self.risk_assymetry_factor = risk_assymetry_factor

    def cvae_parameters(self, recurse: bool = True):
        yield from super().cvae_parameters(recurse)
        yield from self.future_encoder.parameters(recurse)

    def get_parameters(self, recurse: bool = True):
        """Returns a list of two parameter iterators: cvae and encoder only."""
        return [
            self.cvae_parameters(recurse),
            self.biased_parameters(recurse),
        ]

    def set_training_mode(self, training_mode: str) -> None:
        """
        Change the training mode (get_loss function will be different depending on the mode).

        Warning: This does not freeze the decoder because the gradient must pass through it.
            The decoder should be frozen at the optimizer level when changing mode.
        """
        assert training_mode in ["cvae", "bias"]
        self.training_mode = training_mode
        if training_mode == "cvae":
            self.get_loss = self.get_loss_cvae
        else:
            self.get_loss = self.get_loss_biased

    def forward_future(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        y: torch.Tensor,
        mask_y: torch.Tensor,
        offset: torch.Tensor,
        return_inference: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, AbstractLatentDistribution],
        Tuple[torch.Tensor, AbstractLatentDistribution, AbstractLatentDistribution],
    ]:
        """Forward function that outputs a noisy reconstruction of y and parameters of latent
        posterior distribution

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor of bool mask
            y: (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_y: (batch_size, num_agents, num_steps_future) tensor of bool mask.
            offset: (batch_size, num_agents, state_dim) offset position from ego.
            return_inference: (optional) Set to true if z_mean_inference and z_log_std_inference should be returned, Defaults to None.

        Returns:
            noisy reconstruction y of size (batch_size, num_agents, num_steps_future, state_dim), and the
            distribution of the latent posterior, as well as, optionally, the distribution of the latent inference posterior.
        """

        encoded_map = self.map_encoder(map, mask_map)
        mask_map = mask_map.any(-1)
        encoded_absolute = self.absolute_encoder(offset)

        latent_distribution = self.future_encoder(
            x,
            mask_x,
            y=y,
            mask_y=mask_y,
            encoded_absolute=encoded_absolute,
            encoded_map=encoded_map,
            mask_map=mask_map,
        )
        z_sample, weights = latent_distribution.sample()
        mask_z = mask_x.any(-1)

        y_sample = self.decoder(
            z_sample,
            mask_z,
            x,
            mask_x,
            encoded_absolute,
            encoded_map,
            mask_map,
            offset,
        )

        if return_inference:
            inference_distribution = self.inference_encoder(
                x,
                mask_x,
                encoded_absolute,
                encoded_map,
                mask_map,
            )

            return (
                y_sample,
                latent_distribution,
                inference_distribution,
            )
        else:
            return y_sample, latent_distribution

    def get_loss_cvae(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        y: torch.Tensor,
        *,
        mask_y: torch.Tensor,
        mask_loss: torch.Tensor,
        offset: torch.Tensor,
        unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        kl_weight: float,
        kl_threshold: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute and return risk-biased CVAE loss averaged over batch and sequence time steps,
        along with desired loss-related metrics for logging

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor True where map features are good False where it is padding
            y: (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_y: (batch_size, num_agents, num_steps_future) tensor of bool mask.
            mask_loss: (batch_size, num_agents, num_steps_future) tensor of bool mask set to True where the loss
                should be computed and to False where it shouldn't
            offset : (batch_size, num_agents, state_dim) offset position from ego.
            unnormalizer: function that takes in a trajectory and an offset and that outputs the
                unnormalized trajectory
            kl_weight: weight to apply to the KL loss (normal value is 1.0, larger values can be
                used for disentanglement)
            kl_threshold: minimum float value threshold applied to the KL loss

        Returns:
            torch.Tensor: (1,) loss tensor
            dict: dict that contains loss-related metrics to be logged
        """
        log_dict = dict()

        if not mask_loss.any():
            warn("A batch is dropped because the whole loss is masked.")
            return torch.zeros(1, requires_grad=True), {}

        mask_z = mask_x.any(-1)
        # sum_mask_z = mask_z.float().sum().clamp_min(1)

        (y_sample, latent_distribution, inference_distribution) = self.forward_future(
            x,
            mask_x,
            map,
            mask_map,
            y,
            mask_y,
            offset,
            return_inference=True,
        )

        # sum_mask_z *= latent_distribution.mu.shape[-1]

        # log_dict["latent/abs_mean"] = (
        #     (latent_distribution.mu.abs() * mask_z.unsqueeze(-1).float()).sum() / sum_mask_z
        # ).item()
        # log_dict["latent/std"] = (
        #     (latent_distribution.logvar.exp() * mask_z.unsqueeze(-1).float()).sum() / sum_mask_z
        # ).item()
        log_dict["fde/encoded"] = FDE(
            unnormalizer(y_sample, offset), unnormalizer(y, offset), mask_loss
        ).item()
        rec_loss = reconstruction_loss(y_sample, y, mask_loss)

        kl_loss = latent_distribution.kl_loss(
            inference_distribution,
            kl_threshold,
            mask_z,
        )

        # self.prior_distribution.to(latent_distribution.mu.device)

        kl_loss_prior = latent_distribution.kl_loss(
            self.prior_distribution,
            kl_threshold,
            mask_z,
        )

        sampling_loss = latent_distribution.sampling_loss()

        log_dict["loss/rec"] = rec_loss.item()
        log_dict["loss/kl"] = kl_loss.item()
        log_dict["loss/kl_prior"] = kl_loss_prior.item()
        log_dict["loss/sampling"] = sampling_loss.item()
        log_dict.update(latent_distribution.log_dict("future"))
        log_dict.update(inference_distribution.log_dict("inference"))

        loss = (
            rec_loss
            + kl_weight * kl_loss
            + self.regularization_factor * kl_loss_prior
            + sampling_loss
        )

        log_dict["loss/total"] = loss.item()

        return loss, log_dict

    def get_loss_biased(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        y: torch.Tensor,
        *,
        mask_loss: torch.Tensor,
        offset: torch.Tensor,
        unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        risk_level: torch.Tensor,
        x_ego: torch.Tensor,
        y_ego: torch.Tensor,
        kl_weight: float,
        kl_threshold: float,
        risk_weight: float,
        n_samples_risk: int,
        n_samples_biased: int,
        dt: float,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute and return risk-biased CVAE loss averaged over batch and sequence time steps,
        along with desired loss-related metrics for logging

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor True where map features are good False where it is padding
            y: (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_loss: (batch_size, num_agents, num_steps_future) tensor of bool mask set to True where the loss
                should be computed and to False where it shouldn't
            offset : (batch_size, num_agents, state_dim) offset position from ego.
            unnormalizer: function that takes in a trajectory and an offset and that outputs the
                unnormalized trajectory
            risk_level: (batch_size, num_agents) tensor of risk levels desired for future trajectories
            x_ego: (batch_size, 1, num_steps, state_dim) tensor of ego history
            y_ego: (batch_size, 1, num_steps_future, state_dim) tensor of ego future trajectory
            kl_weight: weight to apply to the KL loss (normal value is 1.0, larger values can be
                used for disentanglement)
            kl_threshold: minimum float value threshold applied to the KL loss
            risk_weight: weight to apply to the risk loss (beta parameter in our document)
            n_samples_risk: number of sample to use for Monte-Carlo estimation of the risk using the unbiased distribution
            n_samples_biased: number of sample to use for Monte-Carlo estimation of the risk using the biased distribution
            dt: time step in trajectories

        Returns:
            torch.Tensor: (1,) loss tensor
            dict: dict that contains loss-related metrics to be logged
        """
        log_dict = dict()

        if not mask_loss.any():
            warn("A batch is dropped because the whole loss is masked.")
            return torch.zeros(1, requires_grad=True), {}

        mask_z = mask_x.any(-1)

        # Computing unbiased samples
        n_samples_risk = max(1, n_samples_risk)
        n_samples_biased = max(1, n_samples_biased)
        cost = []
        weights = []
        pack_size = min(n_samples_risk, n_samples_biased)
        with torch.no_grad():
            encoded_map = self.map_encoder(map, mask_map)
            mask_map = mask_map.any(-1)
            encoded_absolute = self.absolute_encoder(offset)

            inference_distribution = self.inference_encoder(
                x,
                mask_x,
                encoded_absolute,
                encoded_map,
                mask_map,
            )
            for _ in range(n_samples_risk // pack_size):
                z_samples, w = inference_distribution.sample(
                    n_samples=pack_size,
                )

                y_samples = self.decoder(
                    z_samples=z_samples,
                    mask_z=mask_z,
                    x=x,
                    mask_x=mask_x,
                    encoded_absolute=encoded_absolute,
                    encoded_map=encoded_map,
                    mask_map=mask_map,
                    offset=offset,
                )

                mask_loss_samples = repeat(mask_loss, "b a t -> b a s t", s=pack_size)
                # Computing unbiased cost
                cost.append(
                    get_cost(
                        self._cost,
                        x,
                        y_samples,
                        offset,
                        x_ego,
                        y_ego,
                        dt,
                        unnormalizer,
                        mask_loss_samples,
                    )
                )
                weights.append(w)

            cost = torch.cat(cost, 2)
            weights = torch.cat(weights, 2)
            risk_cost = self._risk(risk_level, cost, weights)

            log_dict["fde/prior"] = FDE(
                unnormalizer(y_samples, offset),
                unnormalizer(y, offset).unsqueeze(-3),
                mask_loss_samples,
            ).item()

        mask_cost_samples = repeat(mask_z, "b a -> b a s", s=n_samples_risk)
        mean_cost = (cost * mask_cost_samples.float() * weights).sum(2) / (
            (mask_cost_samples.float() * weights).sum(2).clamp_min(1)
        )
        log_dict["cost/mean"] = (
            (mean_cost * mask_loss.any(-1).float()).sum()
            / (mask_loss.any(-1).float().sum())
        ).item()

        # Computing biased latent parameters
        biased_distribution = self.biased_encoder(
            x,
            mask_x,
            encoded_absolute.detach(),
            encoded_map.detach(),
            mask_map,
            risk_level=risk_level,
            x_ego=x_ego,
            y_ego=y_ego,
            offset=offset,
        )
        biased_distribution = inference_distribution.average(
            biased_distribution, risk_level.unsqueeze(-1)
        )

        # sum_mask_z = mask_z.float().sum().clamp_min(1)* biased_distribution.mu.shape[-1]
        # log_dict["latent/abs_mean_biased"] = (
        #     (biased_distribution.mu.abs() * mask_z.unsqueeze(-1).float()).sum() / sum_mask_z
        # ).item()
        # log_dict["latent/var_biased"] = (
        #     (biased_distribution.logvar.exp() * mask_z.unsqueeze(-1).float()).sum() / sum_mask_z
        # ).item()

        # Computing biased samples
        z_biased_samples, weights = biased_distribution.sample(
            n_samples=n_samples_biased
        )
        mask_z_samples = repeat(mask_z, "b a -> b a s ()", s=n_samples_biased)
        log_dict["latent/abs_samples_biased"] = (
            (z_biased_samples.abs() * mask_z_samples.float()).sum()
            / (mask_z_samples.float().sum())
        ).item()

        y_biased_samples = self.decoder(
            z_samples=z_biased_samples,
            mask_z=mask_z,
            x=x,
            mask_x=mask_x,
            encoded_absolute=encoded_absolute,
            encoded_map=encoded_map,
            mask_map=mask_map,
            offset=offset,
        )

        log_dict["fde/prior_biased"] = FDE(
            unnormalizer(y_biased_samples, offset),
            unnormalizer(y, offset).unsqueeze(2),
            mask_loss=mask_loss_samples,
        ).item()

        # Computing biased cost
        biased_cost = get_cost(
            self._cost,
            x,
            y_biased_samples,
            offset,
            x_ego,
            y_ego,
            dt,
            unnormalizer,
            mask_loss_samples,
        )
        mask_cost_samples = mask_z_samples.squeeze(-1)
        mean_biased_cost = (biased_cost * mask_cost_samples.float() * weights).sum(
            2
        ) / ((mask_cost_samples.float() * weights).sum(2).clamp_min(1))
        log_dict["cost/mean_biased"] = (
            (mean_biased_cost * mask_loss.any(-1).float()).sum()
            / (mask_loss.any(-1).float().sum())
        ).item()

        log_dict["cost/risk"] = (
            (risk_cost * mask_loss.any(-1).float()).sum()
            / (mask_loss.any(-1).float().sum())
        ).item()

        # Computing loss between risk and biased cost
        risk_loss = risk_loss_function(
            mean_biased_cost,
            risk_cost.detach(),
            mask_loss.any(-1),
            self.risk_assymetry_factor,
        )
        log_dict["loss/risk"] = risk_loss.item()

        # Computing KL loss between prior and biased latent
        kl_loss = inference_distribution.kl_loss(
            biased_distribution,
            kl_threshold,
            mask_z=mask_z,
        )
        log_dict["loss/kl"] = kl_loss.item()

        loss = risk_weight * risk_loss + kl_weight * kl_loss
        log_dict["loss/total"] = loss.item()

        log_dict["loss/risk_weight"] = risk_weight
        log_dict.update(inference_distribution.log_dict("inference"))
        log_dict.update(biased_distribution.log_dict("biased"))

        return loss, log_dict

    def get_prediction_accuracy(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        y: torch.Tensor,
        mask_loss: torch.Tensor,
        x_ego: torch.Tensor,
        y_ego: torch.Tensor,
        offset: torch.Tensor,
        unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        risk_level: torch.Tensor,
        num_samples_min_fde: int = 0,
    ) -> dict:
        """
        A function that calls the predict method and returns a dict that contains prediction
        metrics, which measure accuracy with respect to ground-truth future trajectory y
        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor True where map features are good False where it is padding
            y: (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_loss: (batch_size, num_agents, num_steps_future) tensor of bool mask set to True where the loss
                should be computed and to False where it shouldn't
            x_ego: (batch_size, 1, num_steps, state_dim) tensor of ego history
            y_ego: (batch_size, 1, num_steps_future, state_dim) tensor of ego future trajectory
            offset: (batch_size, num_agents, state_dim) offset position from ego

            unnormalizer: function that takes in a trajectory and an offset and that outputs the
                          unnormalized trajectory
            risk_level: (batch_size, num_agents) tensor of risk levels desired for future trajectories
            num_samples_min_fde: number of samples to use when computing the minimum final displacement error
        Returns:
            dict: dict that contains prediction-related metrics to be logged
        """
        log_dict = dict()
        with torch.no_grad():
            batch_size = x.shape[0]
            beg = 0
            y_predict = []

            # Limit the batch size so the num_samples_min_fde value does not impact the memory usage
            for i in range(batch_size // num_samples_min_fde + 1):
                sub_batch_size = num_samples_min_fde
                end = beg + sub_batch_size

                y_predict.append(
                    unnormalizer(
                        self.forward(
                            x=x[beg:end],
                            mask_x=mask_x[beg:end],
                            map=map[beg:end],
                            mask_map=mask_map[beg:end],
                            offset=offset[beg:end],
                            x_ego=x_ego[beg:end],
                            y_ego=y_ego[beg:end],
                            risk_level=None,
                            n_samples=num_samples_min_fde,
                        )[0],
                        offset[beg:end],
                    )
                )
                beg = end
                if beg >= batch_size:
                    break

            # Limit the batch size so the num_samples_min_fde value does not impact the memory usage
            if risk_level is not None:
                y_predict_biased = []
                beg = 0
                for i in range(batch_size // num_samples_min_fde + 1):
                    sub_batch_size = num_samples_min_fde
                    end = beg + sub_batch_size
                    y_predict_biased.append(
                        unnormalizer(
                            self.forward(
                                x=x[beg:end],
                                mask_x=mask_x[beg:end],
                                map=map[beg:end],
                                mask_map=mask_map[beg:end],
                                offset=offset[beg:end],
                                x_ego=x_ego[beg:end],
                                y_ego=y_ego[beg:end],
                                risk_level=risk_level[beg:end],
                                n_samples=num_samples_min_fde,
                            )[0],
                            offset[beg:end],
                        )
                    )
                    beg = end
                    if beg >= batch_size:
                        break
                y_predict_biased = torch.cat(y_predict_biased, 0)
                if num_samples_min_fde > 0:
                    repeated_mask_loss = repeat(
                        mask_loss, "b a t -> b a samples t", samples=num_samples_min_fde
                    )
                    log_dict["fde/prior_biased"] = FDE(
                        y_predict_biased, y.unsqueeze(-3), mask_loss=repeated_mask_loss
                    ).item()
                    log_dict["minfde/prior_biased"] = minFDE(
                        y_predict_biased, y.unsqueeze(-3), mask_loss=repeated_mask_loss
                    ).item()
                else:
                    log_dict["fde/prior_biased"] = FDE(
                        y_predict_biased, y, mask_loss=mask_loss
                    ).item()

            y_predict = torch.cat(y_predict, 0)
            y_unnormalized = unnormalizer(y, offset)
        if num_samples_min_fde > 0:
            repeated_mask_loss = repeat(
                mask_loss, "b a t -> b a samples t", samples=num_samples_min_fde
            )
            log_dict["fde/prior"] = FDE(
                y_predict, y_unnormalized.unsqueeze(-3), mask_loss=repeated_mask_loss
            ).item()
            log_dict["minfde/prior"] = minFDE(
                y_predict, y_unnormalized.unsqueeze(-3), mask_loss=repeated_mask_loss
            ).item()
        else:
            log_dict["fde/prior"] = FDE(
                y_predict, y_unnormalized, mask_loss=mask_loss
            ).item()
        return log_dict


def cvae_factory(
    params: CVAEParams,
    cost_function: BaseCostTorch,
    risk_estimator: AbstractMonteCarloRiskEstimator,
    training_mode: str = "cvae",
):
    """Biased CVAE with a biased MLP encoder and an MLP decoder
    Args:
        params: dataclass defining the necessary parameters
        cost_function: cost function used to compute the risk objective
        risk_estimator: risk estimator used to compute the risk objective
        training_mode: "inference", "cvae" or "bias" set what is the training mode
        latent_distribution: "gaussian" or "quantized" set the latent distribution
    """

    absolute_encoder_nn = MLP(
        params.dynamic_state_dim,
        params.hidden_dim,
        params.hidden_dim,
        params.num_hidden_layers,
        params.is_mlp_residual,
    )

    map_encoder_nn = MapEncoderNN(params)

    if params.latent_distribution == "gaussian":
        latent_distribution_creator = GaussianLatentDistribution
        prior_distribution = GaussianLatentDistribution(
            torch.zeros(1, 1, 2 * params.latent_dim)
        )
        future_encoder_latent_dim = 2 * params.latent_dim
        inference_encoder_latent_dim = 2 * params.latent_dim
        biased_encoder_latent_dim = 2 * params.latent_dim
    elif params.latent_distribution == "quantized":
        latent_distribution_creator = QuantizedDistributionCreator(
            params.latent_dim, params.num_vq
        )
        prior_distribution = latent_distribution_creator(
            torch.zeros(1, 1, params.num_vq)
        )
        future_encoder_latent_dim = params.latent_dim
        inference_encoder_latent_dim = params.num_vq
        biased_encoder_latent_dim = params.num_vq

    biased_encoder_nn = BiasedEncoderNN(
        params,
        biased_encoder_latent_dim,
        num_steps=params.num_steps,
    )
    biased_encoder = CVAEEncoder(
        biased_encoder_nn, latent_distribution_creator=latent_distribution_creator
    )

    future_encoder_nn = FutureEncoderNN(
        params, future_encoder_latent_dim, params.num_steps + params.num_steps_future
    )
    future_encoder = CVAEEncoder(
        future_encoder_nn, latent_distribution_creator=latent_distribution_creator
    )

    inference_encoder_nn = InferenceEncoderNN(
        params, inference_encoder_latent_dim, params.num_steps
    )
    inference_encoder = CVAEEncoder(
        inference_encoder_nn, latent_distribution_creator=latent_distribution_creator
    )

    decoder_nn = DecoderNN(params)
    decoder = CVAEAccelerationDecoder(decoder_nn)
    # decoder = CVAEParametrizedDecoder(decoder_nn)

    if training_mode == "inference":
        cvae = InferenceBiasedCVAE(
            absolute_encoder_nn,
            map_encoder_nn,
            biased_encoder,
            inference_encoder,
            decoder,
            prior_distribution=prior_distribution,
        )
        cvae.eval()
        return cvae
    else:
        return TrainingBiasedCVAE(
            absolute_encoder_nn,
            map_encoder_nn,
            biased_encoder,
            inference_encoder,
            decoder,
            future_encoder=future_encoder,
            cost_function=cost_function,
            risk_estimator=risk_estimator,
            training_mode=training_mode,
            latent_regularization=params.latent_regularization,
            risk_assymetry_factor=params.risk_assymetry_factor,
            prior_distribution=prior_distribution,
        )
