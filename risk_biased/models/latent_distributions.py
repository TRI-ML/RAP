from typing import Optional, Callable, Tuple
import warnings

from abc import ABC, abstractmethod
from einops import rearrange, repeat
import torch
import torch.nn as nn


def relaxed_one_hot_categorical_without_replacement(temperature, logits, num_samples=1):
    # See paper Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement (https://arxiv.org/pdf/1903.06059.pdf)
    # for explanation of the trick
    scores = (
        (torch.distributions.Gumbel(logits, 1).rsample() / temperature)
        .softmax(-1)
        .clamp_min(1e-10)
    )
    top_scores, top_indices = torch.topk(
        scores,
        num_samples,
        dim=-1,
    )
    return scores, top_indices


class AbstractLatentDistribution(nn.Module, ABC):
    """Base class for latent distribution"""

    @abstractmethod
    def sample(
        self, num_samples: int, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the latent distribution."""

    @abstractmethod
    def kl_loss(
        self,
        other: "GaussianLatentDistribution",
        threshold: float = 0,
        mask_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the KL divergence between two latent distributions."""

    @abstractmethod
    def sampling_loss(self) -> torch.Tensor:
        """Loss of the latent distribution."""

    @abstractmethod
    def average(
        self, other: "AbstractLatentDistribution", weight_other: torch.Tensor
    ) -> "AbstractLatentDistribution":
        """Average of the latent distribution."""

    @abstractmethod
    def log_dict(self, type: str) -> dict:
        """Log the latent distribution values."""


class GaussianLatentDistribution(AbstractLatentDistribution):
    """Gaussian latent distribution"""

    def __init__(self, latent_representation: torch.Tensor):
        super().__init__()
        mu, logvar = torch.chunk(latent_representation, 2, dim=-1)
        self.register_buffer("mu", mu, False)
        self.register_buffer("logvar", logvar, False)

    def sample(
        self, n_samples: int = 0, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from Gaussian with a reparametrization trick

        Args:
            n_samples (optional): number of samples to make, (if 0 one sample with no extra
                dimension). Defaults to 0.
        Returns:
            Random Gaussian sample of size (some_shape, (n_samples), latent_dim)
        """

        std = (self.logvar / 2).exp()
        if n_samples <= 0:
            eps = torch.randn_like(std)
            latent_samples = self.mu + eps * std
            weights = torch.ones_like(latent_samples[..., 0])
        else:
            eps = torch.randn(
                [*std.shape[:-1], n_samples, self.mu.shape[-1]], device=std.device
            )
            # Reshape
            latent_samples = self.mu.unsqueeze(-2) + eps * std.unsqueeze(-2)
            weights = torch.ones_like(latent_samples[..., 0]) / n_samples
        return latent_samples, weights

    def kl_loss(
        self,
        other: "GaussianLatentDistribution",
        threshold: float = 0,
        mask_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the KL divergence between two latent distributions."""
        assert type(other) == GaussianLatentDistribution
        kl_loss = (
            (
                other.logvar
                - self.logvar
                + ((self.mu - other.mu).square() + self.logvar.exp())
                / other.logvar.exp()
                - 1
            )
            * 0.5
        ).clamp_min(threshold)
        if mask_z is None:
            return kl_loss.mean()
        else:
            assert mask_z.any()
            return torch.sum(kl_loss.mean(-1) * mask_z) / torch.sum(mask_z)

    def sampling_loss(self) -> torch.Tensor:
        return torch.zeros(1, device=self.mu.device)

    def average(
        self, other: "GaussianLatentDistribution", weight_other: torch.Tensor
    ) -> "GaussianLatentDistribution":
        assert type(other) == GaussianLatentDistribution
        assert other.mu.shape == self.mu.shape
        average_log_var = (
            self.logvar.exp() * (1 - weight_other) + other.logvar.exp() * weight_other
        ).log()
        return GaussianLatentDistribution(
            torch.cat(
                (
                    self.mu * (1 - weight_other) + other.mu * weight_other,
                    average_log_var,
                ),
                dim=-1,
            )
        )

    def log_dict(self, type: str) -> dict:
        return {
            f"latent/{type}/abs_mean": self.mu.abs().mean(),
            f"latent/{type}/std": (self.logvar * 0.5).exp().mean(),
        }


class QuantizedLatentDistribution(AbstractLatentDistribution):
    """Quantized latent distribution.
    It is defined with a codebook of quantized latents and a continuous latent.
    The distribution is based on distances of the continuous latent to the codebook.
    Sampling is only quantizing the continuous latent.

    Args:
        continuous_latent : Continuous latent representation of shape (some_shape, latent_dim)
        codebook : Codebook of shape (num_embeddings, latent_dim)
    """

    def __init__(
        self,
        continuous_latent: torch.Tensor,
        codebook: torch.Tensor,
        flush_weights: Callable[[], None],
        get_weights: Callable[[], torch.Tensor],
        index_add_one_weights: Callable[[torch.Tensor], None],
    ):
        super().__init__()
        self.register_buffer("continuous_latent", continuous_latent, False)
        self.register_buffer("codebook", codebook, False)
        self.flush_weights = flush_weights
        self.get_weights = get_weights
        self.index_add_one_weights = index_add_one_weights
        self.quantization_loss = None
        self.accuracy = None

    def sample(
        self, n_samples: int = 0, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize the continuous latent from the latent dictionary.

        Args:
            latent: (batch_size, num_agents, latent_dim) Continuous latent input

        Returns:
            quantized_latent, quantization_loss
        """
        assert n_samples == 0, "Only one sample is supported for quantized latent"

        distances_to_quantized = (
            (
                self.codebook.view(1, 1, *self.codebook.shape)
                - self.continuous_latent.unsqueeze(-2)
            )
            .square()
            .sum(-1)
        )
        batch_size, num_agents, num_vq = distances_to_quantized.shape

        self.soft_one_hot = (
            (-100 * distances_to_quantized)
            .softmax(dim=-1)
            .view(batch_size, num_agents, num_vq)
        )
        # quantized, args_selected = self.sample(soft_one_hot)
        _, args_selected = torch.min(distances_to_quantized, dim=-1)
        quantized = self.codebook[args_selected, :]
        args_selected = args_selected.view(-1)

        # Update weights
        self.index_add_one_weights(args_selected)

        distances_to_quantized = distances_to_quantized.view(
            batch_size * num_agents, num_vq
        )

        # Resample useless latent vectors
        random_latents = self.continuous_latent.view(
            batch_size * num_agents, self.codebook.shape[-1]
        )[torch.randint(batch_size * num_agents, (num_vq,))]
        codebook_weights = self.get_weights()
        total_samples = codebook_weights.sum()
        # TODO: The value 100 is arbitrary, should it be a parameter?
        # The uselessness of a codebook vector is defined by the number of times it has been sampled
        # if it has been sampled less than 1% of the time, it is pushed towards a random continuous latent sample
        # this prevents the codebook from being dominated by a few vectors
        self.uselessness = (
            (
                torch.where(
                    (codebook_weights < total_samples / (100 * num_vq)).unsqueeze(-1),
                    random_latents.detach() - self.codebook,
                    torch.zeros_like(self.codebook),
                ).abs()
                + 1
            )
            .log()
            .sum(-1)
            .mean()
        )
        # TODO: The value 1e6 is arbitrary, should it be a parameter?
        if total_samples > 1e6 * num_vq:
            # Flush the codebook weights when the number of samples is too high
            # This prevents the codebook from being dominated by its history
            # if a few vectors were visited a lot and also prevents overflows
            self.flush_weights()

        # commit_loss = (self.continuous_latent - quantized.detach()).square().clamp_min(self.distance_threshold).sum(-1).mean()

        self.quantization_loss = (
            (self.continuous_latent - quantized).square().sum(-1).mean()
        )

        quantized = (
            quantized.detach()
            + self.continuous_latent
            - self.continuous_latent.detach()
        )

        self.latent_diversity = (
            (self.continuous_latent[None, ...] - self.continuous_latent[:, None, ...])
            .square()
            .sum(-1)
            .mean()
        )

        return quantized, torch.ones_like(quantized[..., 0]) / num_vq

    def kl_loss(
        self,
        other: "ClassifiedLatentDistribution",
        threshold: float = 0,
        mask_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross entropy between two latent distributions."""
        assert type(other) == ClassifiedLatentDistribution
        min_logits = -10
        max_logits = 10
        pred_log = other.logits.clamp(min_logits, max_logits).log_softmax(-1)
        self_pred = self.soft_one_hot
        self.accuracy = (self_pred.argmax(-1) == other.logits.argmax(-1)).float().mean()
        return -2 * (pred_log * self_pred).sum(-1).mean()

    def sampling_loss(self) -> torch.Tensor:
        if self.quantization_loss is None:
            self.sample()
        return 0.5 * (
            self.quantization_loss + self.uselessness + 0.001 * self.latent_diversity
        )

    def average(
        self, other: "QuantizedLatentDistribution", weight_other: torch.Tensor
    ) -> "QuantizedLatentDistribution":
        raise NotImplementedError(
            "Average is not implemented for QuantizedLatentDistribution"
        )

    def log_dict(self, type: str) -> dict:
        log_dict = {
            f"latent/{type}/quantization_loss": self.quantization_loss,
            f"latent/{type}/uselessness": self.uselessness,
            f"latent/{type}/latent_diversity": self.latent_diversity,
            f"latent/{type}/codebook_abs_mean": self.codebook.abs().mean(),
            f"latent/{type}/codebook_std": self.codebook.std(),
            f"latent/{type}/latent_abs_mean": self.continuous_latent.abs().mean(),
            f"latent/{type}/latent_std": self.continuous_latent.std(),
        }
        if self.accuracy is not None:
            log_dict[f"latent/{type}/accuracy"] = self.accuracy
        return log_dict


class ClassifiedLatentDistribution(AbstractLatentDistribution):
    """Classified latent distribution.
    It is defined with a codebook of quantized latents and a probability distribution over the codebook elements.

    Args:
        logits : Logits of shape (some_shape, num_embeddings)
        codebook : Codebook of shape (num_embeddings, latent_dim)
    """

    def __init__(self, logits: torch.Tensor, codebook: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits, persistent=False)
        self.register_buffer("codebook", codebook, persistent=False)

    def sample(
        self, n_samples: int = 0, replacement: bool = True, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_agents, num_vq = self.logits.shape
        squeeze_out = False
        if n_samples == 0:
            squeeze_out = True
            n_samples = 1
        elif n_samples > self.codebook.shape[0]:
            warnings.warn(
                f"Requested {n_samples} samples but only {self.codebook.shape[0]} are available in the descrete latent space. Switching to replacement=True to support it."
            )
            replacement = True

        if self.training:
            # TODO: should we make the temperature a parameter?
            all_weights, indices = relaxed_one_hot_categorical_without_replacement(
                logits=self.logits, temperature=1, num_samples=n_samples
            )
            selected_latents = self.codebook[indices, :]
            # Cumulative mask of indices that have been sampled in order of probability
            mask_selection = torch.nn.functional.one_hot(indices, num_vq).cumsum(-2)
            mask_selection[..., 1:, :] = mask_selection[..., :-1, :]
            mask_selection[..., 0, :] = 0.0
            # Remove the probability of previous samples to account for sampling without replacement
            masked_weights = all_weights.unsqueeze(-2) * (1 - mask_selection.float())
            # Renormalize the probabilities to sum to 1
            masked_weights = masked_weights / masked_weights.sum(-1, keepdim=True)

            latent_samples = (
                masked_weights.unsqueeze(-1)
                * self.codebook[None, None, None, ...].detach()
            ).sum(-2)
            latent_samples = (
                selected_latents.detach() + latent_samples - latent_samples.detach()
            )
            probs = torch.gather(self.logits.softmax(-1), -1, indices)
        else:
            probs = self.logits.softmax(-1)
            samples = torch.multinomial(
                probs.view(batch_size * num_agents, num_vq),
                n_samples,
                replacement=replacement,
            )
            latent_samples = self.codebook[samples]
            probs = torch.gather(
                probs, -1, samples.view(batch_size, num_agents, num_vq)
            )

        if squeeze_out:
            latent_samples = latent_samples.view(
                batch_size, num_agents, self.codebook.shape[-1]
            )
        else:
            latent_samples = latent_samples.view(
                batch_size, num_agents, n_samples, self.codebook.shape[-1]
            )
        return latent_samples, probs

    def kl_loss(
        self,
        other: "ClassifiedLatentDistribution",
        threshold: float = 0,
        mask_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross entropy between two latent distributions. Self being the reference distribution and other the distribution to compare."""
        assert type(other) == ClassifiedLatentDistribution
        min_logits = -10
        max_logits = 10
        pred_log = other.logits.clamp(min_logits, max_logits).log_softmax(-1)
        self_pred = (
            (0.5 * (self.logits.detach() + self.logits))
            .clamp(min_logits, max_logits)
            .softmax(-1)
        )
        return -2 * (pred_log * self_pred).sum(-1).mean()

    def sampling_loss(self) -> torch.Tensor:
        return torch.zeros(1, device=self.logits.device)

    def average(
        self, other: "ClassifiedLatentDistribution", weight_other: torch.Tensor
    ) -> "ClassifiedLatentDistribution":
        assert type(other) == ClassifiedLatentDistribution
        assert (self.codebook == other.codebook).all()
        return ClassifiedLatentDistribution(
            (
                self.logits.exp() * (1 - weight_other)
                + other.logits.exp() * weight_other
            ).log(),
            self.codebook,
        )

    def log_dict(self, type: str) -> dict:
        max_probs, _ = self.logits.softmax(-1).max(-1)
        return {
            f"latent/{type}/codebook_abs_mean": self.codebook.abs().mean(),
            f"latent/{type}/codebook_std": self.codebook.std(),
            f"latent/{type}/class_max_mean": max_probs.mean(),
            f"latent/{type}/class_max_std": max_probs.std(),
        }


class QuantizedDistributionCreator(nn.Module):
    """Creates a distribution from a latent vector."""

    def __init__(
        self,
        latent_dim: int,
        num_embeddings: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.codebook = nn.Parameter(torch.randn(num_embeddings, latent_dim))
        self.register_buffer(
            "codebook_weights",
            torch.ones(num_embeddings, requires_grad=False),
            persistent=False,
        )

    def _flush_codebook_weights(self):
        self.codebook_weights = torch.ones_like(self.codebook_weights)

    def _get_codebook_weights(self):
        return self.codebook_weights

    def _index_add_one_codebook_weight(self, indices: torch.Tensor):
        self.codebook_weights = self.codebook_weights.index_add(
            0,
            indices.flatten(),
            torch.ones_like(self.codebook_weights[indices]),
        )

    def forward(self, latent: torch.Tensor) -> AbstractLatentDistribution:
        if latent.shape[-1] == self.latent_dim:
            return QuantizedLatentDistribution(
                latent,
                self.codebook,
                self._flush_codebook_weights,
                self._get_codebook_weights,
                self._index_add_one_codebook_weight,
            )
        elif latent.shape[-1] == self.num_embeddings:
            return ClassifiedLatentDistribution(
                latent,
                self.codebook,
            )
        else:
            raise ValueError(f"Latent vector has wrong dimension: {latent.shape[-1]}")
