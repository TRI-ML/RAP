import inspect
import math
import warnings
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class AbstractMonteCarloRiskEstimator(ABC):
    """Abstract class for Monte Carlo estimation of risk objectives"""

    @abstractmethod
    def __call__(self, risk_level: Tensor, cost: Tensor) -> Tensor:
        """Computes and returns the risk objective estimated on the cost tensor

        Args:
            risk_level: (batch_size,) tensor of risk-level at which the risk objective is computed
            cost: (batch_size, n_samples) tensor of cost samples
        Returns:
            risk tensor of size (batch_size,)
        """


class EntropicRiskEstimator(AbstractMonteCarloRiskEstimator):
    """Monte Carlo estimator for the entropic risk objective.
    This estimator computes the entropic risk as 1/risk_level * log( mean( exp(risk_level * cost), 1))
    However, this is unstable.
    When risk_level is large, the logsumexp trick is used.
    When risk_level is small, it computes entropic_risk for small values of risk_level as the second order Taylor expansion instead.

    Args:
        eps: Risk-level threshold to switch between logsumexp and Taylor expansion. Defaults to 1e-4.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = eps

    def __call__(self, risk_level: Tensor, cost: Tensor, weights: Tensor) -> Tensor:
        """Computes and returns the entropic risk estimated on the cost tensor

        Args:
            risk_level: (batch_size, n_agents,) tensor of risk-level at which the risk objective is computed
            cost: (batch_size, n_agents, n_samples) cost tensor
            weights: (batch_size, n_agents, n_samples) tensor of weights for the cost samples
        Returns:
            entropic risk tensor of size (batch_size,)
        """
        weights = weights / weights.sum(dim=-1, keepdim=True)
        batch_size, n_agents, n_samples = cost.shape
        entropic_risk_cost_large_sigma = (
            ((risk_level.view(batch_size, n_agents, 1) * cost).exp() * weights)
            .sum(-1)
            .log()
        ) / risk_level

        mean = (cost * weights).sum(dim=-1)
        var = (cost**2 * weights).sum(dim=-1) - mean**2

        var, mean = torch.var_mean(cost, -1)
        entropic_risk_cost_small_sigma = mean + 0.5 * risk_level * var

        return torch.where(
            torch.abs(risk_level) > self.eps,
            entropic_risk_cost_large_sigma,
            entropic_risk_cost_small_sigma,
        )


class CVaREstimator(AbstractMonteCarloRiskEstimator):
    """Monte Carlo estimator for the conditional value-at-risk objective.
    This estimator is proposed in the following references, and shown to be consistent.
    - Hong et al. (2014), "Monte Carlo Methods for Value-at-Risk and Conditional Value-at-Risk: A Review"
    - Traindade et al. (2007), "Financial prediction with constrained tail risk"
    When risk_level is larger than 1 - eps, it falls back to the max operator

    Args:
        Args:
        eps: Risk-level threshold to switch between CVaR and Max. Defaults to 1e-4.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = eps

    def __call__(self, risk_level: Tensor, cost: Tensor, weights: Tensor) -> Tensor:
        """Computes and returns the conditional value-at-risk estimated on the cost tensor

        Args:
            risk_level: (batch_size, n_agents) tensor of risk-level in [0, 1] at which the CVaR risk is computed
            cost: (batch_size, n_agents, n_samples) cost tensor
            weights: (batch_size, n_agents, n_samples) tensor of weights for the cost samples

        Returns:
            conditional value-at-risk tensor of size (batch_size, n_agents)
        """
        assert risk_level.shape[0] == cost.shape[0]
        assert risk_level.shape[1] == cost.shape[1]
        if weights is None:
            weights = torch.ones_like(cost) / cost.shape[-1]
        else:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        if not (torch.all(0.0 <= risk_level) and torch.all(risk_level <= 1.0)):
            warnings.warn(
                "risk_level is defined only between 0.0 and 1.0 for CVaR. Exceeded values will be clamped."
            )
            risk_level = torch.clamp(risk_level, min=0.0, max=1.0)

        cvar_risk_high = cost.max(dim=-1).values

        sorted_indices = torch.argsort(cost, dim=-1)
        # cost_sorted = cost.sort(dim=-1, descending=False).values
        cost_sorted = torch.gather(cost, -1, sorted_indices)
        weights_sorted = torch.gather(weights, -1, sorted_indices)
        idx_to_choose = torch.argmax(
            (weights_sorted.cumsum(dim=-1) >= risk_level.unsqueeze(-1)).float(), -1
        )

        value_at_risk_mc = cost_sorted.gather(-1, idx_to_choose.unsqueeze(-1)).squeeze(
            -1
        )

        # weights_at_risk_mc = 1 - weights_sorted.cumsum(-1).gather(
        #     -1, idx_to_choose.unsqueeze(-1)
        # ).squeeze(-1)
        # cvar_risk_mc = value_at_risk_mc + (
        #     (torch.relu(cost - value_at_risk_mc.unsqueeze(-1)) * weights).sum(dim=-1)
        #     / weights_at_risk_mc
        # )
        # cvar = torch.where(weights_at_risk_mc < self.eps, cvar_risk_high, cvar_risk_mc)

        cvar_risk_mc = value_at_risk_mc + 1 / (1 - risk_level) * (
            (torch.relu(cost - value_at_risk_mc.unsqueeze(-1)) * weights).sum(dim=-1)
        )
        cvar = torch.where(risk_level > 1 - self.eps, cvar_risk_high, cvar_risk_mc)
        return cvar


def get_risk_estimator(estimator_params: dict) -> AbstractMonteCarloRiskEstimator:
    """Function that returns the Monte Carlo risk estimator hat matches the given parameters.
    Tries to give a comprehensive feedback if the parameters are not recognized and raise an error.

    Args:
        Risk estimator should be one of the following types (with different parameter values as desired) :
            {"type": "entropic", "eps": 1e-4},
            {"type": "cvar", "eps": 1e-4}

    Raises:
        RuntimeError: If the given parameter dictionary does not match one of the expected formats, raise a comprehensive error.

    Returns:
        A risk estimator matching the given parameters.
    """
    known_types = ["entropic", "cvar"]
    try:
        if estimator_params["type"].lower() == "entropic":
            expected_params = inspect.getfullargspec(EntropicRiskEstimator)[0][1:]
            return EntropicRiskEstimator(estimator_params["eps"])
        elif estimator_params["type"].lower() == "cvar":
            expected_params = inspect.getfullargspec(CVaREstimator)[0][1:]
            return CVaREstimator(estimator_params["eps"])
        else:
            raise RuntimeError(
                f"Risk estimator '{estimator_params['type']}' is unknown. It should be one of {known_types}."
            )
    except KeyError:
        if "type" in estimator_params:
            raise RuntimeError(
                f"""The estimator '{estimator_params['type']}' is known but the given parameters
                {estimator_params} do not match the expected parameters {expected_params}."""
            )
        else:
            raise RuntimeError(
                f"""The given estimator parameters {estimator_params} do not define the estimator
                type in the field 'type'. Please add a field 'type' and set it to one of the
                handeled types: {known_types}."""
            )


class AbstractRiskLevelSampler(ABC):
    """Abstract class for a risk-level sampler for training and evaluating risk-biased predictors"""

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """Returns a tensor of size batch_size with sampled risk-level values

        Args:
            batch_size: number of elements in the out tensor
            device: device of the output tensor

        Returns:
            A tensor of shape(batch_size,) filled with sampled risk values
        """

    @abstractmethod
    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        """Returns a tensor of size batch_size with high values of risk.

        Args:
            batch_size: number of elements in the out tensor
            device: device of the output tensor

        Returns:
            A tensor of shape (batchc_size,) filled with the highest possible risk-level
        """


class UniformRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a uniform distribution

    Args:
        min: minimum risk-level
        max: maximum risk-level
    """

    def __init__(self, min: int, max: int) -> None:
        self.min = min
        self.max = max

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.rand(batch_size, device=device) * (self.max - self.min) + self.min

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, device=device) * self.max


class NormalRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a normal distribution

    Args:
        mean: average risk-level
        sigma: standard deviation of the sampler
    """

    def __init__(self, mean: int, sigma: int) -> None:
        self.mean = mean
        self.sigma = sigma

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randn(batch_size, device=device) * self.sigma + self.mean

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, device=device) * self.sigma * 3


class BernoulliRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a scaled Bernoulli distribution

    Args:
        min: minimum risk-level
        max: maximum risk-level
        p: Bernoulli parameter
    """

    def __init__(self, min: int, max: int, p: int) -> None:
        self.min = min
        self.max = max
        self.p = p

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return (
            torch.bernoulli(torch.ones(batch_size, device=device) * self.p)
            * (self.max - self.min)
            + self.min
        )

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, device=device) * self.max


class BetaRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a scaled Beta distribution

        Distribution properties:
            mean = alpha*(max-min)/(alpha + beta) + min
            mode = (alpha-1)*(max-min)/(alpha + beta - 2) + min
            variance = alpha*beta*(max-min)**2/((alpha+beta)**2*(alpha+beta+1))

    Args:
        min: minimum risk-level
        max: maximum risk-level
        alpha: First distribution parameter
        beta: Second distribution parameter
    """

    def __init__(self, min: int, max: int, alpha: float, beta: float) -> None:
        self.min = min
        self.max = max
        self._distribution = torch.distributions.Beta(
            torch.tensor([alpha], dtype=torch.float32),
            torch.tensor([beta], dtype=torch.float32),
        )

    @property
    def alpha(self):
        return self._distribution.concentration1.item()

    @alpha.setter
    def alpha(self, alpha: float):
        self._distribution = torch.distributions.Beta(
            torch.tensor([alpha], dtype=torch.float32),
            torch.tensor([self.beta], dtype=torch.float32),
        )

    @property
    def beta(self):
        return self._distribution.concentration0.item()

    @beta.setter
    def beta(self, beta: float):
        self._distribution = torch.distributions.Beta(
            torch.tensor([self.alpha], dtype=torch.float32),
            torch.tensor([beta], dtype=torch.float32),
        )

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return (
            self._distribution.sample((batch_size,)).to(device) * (self.max - self.min)
            + self.min
        ).view(batch_size)

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, device=device) * self.max


class Chi2RiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a scaled chi2 distribution

        Distribution properties:
            mean = k*scale + min
            mode = max(k-2, 0)*scale + min
            variance = 2*k*scale**2

    Args:
        min: minimum risk-level
        scale: scaling factor for the risk-level
        k: Chi2 parameter: degrees of freedom of the distribution
    """

    def __init__(self, min: int, scale: float, k: int) -> None:
        self.min = min
        self.scale = scale
        self._distribution = torch.distributions.Chi2(
            torch.tensor([k], dtype=torch.float32)
        )

    @property
    def k(self):
        return self._distribution.df.item()

    @k.setter
    def k(self, k: int):
        self._distribution = torch.distributions.Chi2(
            torch.tensor([k], dtype=torch.float32)
        )

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return (
            self._distribution.sample((batch_size,)).to(device) * self.scale + self.min
        ).view(batch_size)

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        std = self.scale * math.sqrt(2 * self.k)
        return torch.ones(batch_size, device=device) * std * 3


class LogNormalRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a scaled Beta distribution

        Distribution properties:
            mean = exp(mu + sigma**2/2)*scale + min
            mode = exp(mu - sigma**2)*scale + min
            variance = (exp(sigma**2)-1)*exp(2*mu+sigma**2)*scale**2

    Args:
        min: minimum risk-level
        scale: scaling factor for the risk-level
        mu: First distribution parameter
        sigma: maximum risk-level
    """

    def __init__(self, min: int, scale: float, mu: float, sigma: float) -> None:
        self.min = min
        self.scale = scale
        self._distribution = torch.distributions.LogNormal(
            torch.tensor([mu], dtype=torch.float32),
            torch.tensor([sigma], dtype=torch.float32),
        )

    @property
    def mu(self):
        return self._distribution.loc.item()

    @mu.setter
    def mu(self, mu: float):
        self._distribution = torch.distributions.LogNormal(
            torch.tensor([mu], dtype=torch.float32),
            torch.tensor([self.sigma], dtype=torch.float32),
        )

    @property
    def sigma(self) -> float:
        return self._distribution.scale.item()

    @sigma.setter
    def sigma(self, sigma: float):
        self._distribution = torch.distributions.LogNormal(
            torch.tensor([self.mu], dtype=torch.float32),
            torch.tensor([sigma], dtype=torch.float32),
        )

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        return (
            self._distribution.sample((batch_size,)).to(device) * self.scale + self.min
        ).view(batch_size)

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        std = (
            (torch.exp(self.sigma.square()) - 1).sqrt()
            * torch.exp(self.mu + self.sigma.square() / 2)
            * self.scale
        )
        return torch.ones(batch_size, device=device) * 3 * std


class LogUniformRiskLevelSampler(AbstractRiskLevelSampler):
    """Risk-level sampler with a reversed log-uniform distribution (increasing density function). Between min and max.

        Distribution properties:
            mean = (max - min)/ln((max+1)/(min+1)) - 1/scale
            mode = None
            variance = (((max+1)^2 - (min+1)^2)/(2*ln((max+1)/(min+1))) - ((max - min)/ln((max+1)/(min+1)))^2)

    Args:
        min: minimum risk-level
        max: maximum risk-level
        scale: scale to apply to the sampling before applying exponential,
        the output is rescaled back to fit in bounds [min, max] (the higher the scale the less uniform the distribution)
    """

    def __init__(self, min: float, max: float, scale: float) -> None:
        assert min >= 0
        assert max > min
        assert scale > 0
        self.min = min
        self.max = max
        self.scale = scale

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        scale = self.scale / (self.max - self.min)
        max = self.max * scale
        min = self.min * scale
        return (
            max
            - (
                (
                    torch.rand(batch_size, device=device)
                    * (math.log(max + 1) - math.log(min + 1))
                    + math.log(min + 1)
                ).exp()
                - 1
            )
            + min
        ) / scale

    def get_highest_risk(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.ones(batch_size, device=device) * self.max


def get_risk_level_sampler(distribution_params: dict) -> AbstractRiskLevelSampler:
    """Function that returns the risk level sampler that matches the given parameters.
    Tries to give a comprehensive feedback if the parameters are not recognized and raise an error.

    Args:
        Risk distribution should be one of the following types (with different parameter values as desired) :
            {"type": "uniform", "min": 0, "max": 1},
            {"type": "normal", "mean": 0, "sigma": 1},
            {"type": "bernoulli", "p": 0.5, "min": 0, "max": 1},
            {"type": "beta", "alpha": 2, "beta": 5, "min": 0, "max": 1},
            {"type": "chi2", "k": 3, "min": 0, "scale": 1},
            {"type": "log-normal", "mu": 0, "sigma": 1, "min": 0, "scale": 1}
            {"type": "log-uniform", "min": 0, "max": 1, "scale": 1}

    Raises:
        RuntimeError: If the given parameter dictionary does not match one of the expected formats, raise a comprehensive error.

    Returns:
        A risk level sampler matching the given parameters.
    """
    known_types = [
        "uniform",
        "normal",
        "bernoulli",
        "beta",
        "chi2",
        "log-normal",
        "log-uniform",
    ]
    try:
        if distribution_params["type"].lower() == "uniform":
            expected_params = inspect.getfullargspec(UniformRiskLevelSampler)[0][1:]
            return UniformRiskLevelSampler(
                distribution_params["min"], distribution_params["max"]
            )
        elif distribution_params["type"].lower() == "normal":
            expected_params = inspect.getfullargspec(NormalRiskLevelSampler)[0][1:]
            return NormalRiskLevelSampler(
                distribution_params["mean"], distribution_params["sigma"]
            )
        elif distribution_params["type"].lower() == "bernoulli":
            expected_params = inspect.getfullargspec(BernoulliRiskLevelSampler)[0][1:]
            return BernoulliRiskLevelSampler(
                distribution_params["min"],
                distribution_params["max"],
                distribution_params["p"],
            )
        elif distribution_params["type"].lower() == "beta":
            expected_params = inspect.getfullargspec(BetaRiskLevelSampler)[0][1:]
            return BetaRiskLevelSampler(
                distribution_params["min"],
                distribution_params["max"],
                distribution_params["alpha"],
                distribution_params["beta"],
            )
        elif distribution_params["type"].lower() == "chi2":
            expected_params = inspect.getfullargspec(Chi2RiskLevelSampler)[0][1:]
            return Chi2RiskLevelSampler(
                distribution_params["min"],
                distribution_params["scale"],
                distribution_params["k"],
            )
        elif distribution_params["type"].lower() == "log-normal":
            expected_params = inspect.getfullargspec(LogNormalRiskLevelSampler)[0][1:]
            return LogNormalRiskLevelSampler(
                distribution_params["min"],
                distribution_params["scale"],
                distribution_params["mu"],
                distribution_params["sigma"],
            )
        elif distribution_params["type"].lower() == "log-uniform":
            expected_params = inspect.getfullargspec(LogUniformRiskLevelSampler)[0][1:]
            return LogUniformRiskLevelSampler(
                distribution_params["min"],
                distribution_params["max"],
                distribution_params["scale"],
            )
        else:
            raise RuntimeError(
                f"Distribution {distribution_params['type']} is unknown. It should be one of {known_types}."
            )
    except KeyError:
        if "type" in distribution_params:
            raise RuntimeError(
                f"The distribution '{distribution_params['type']}' is known but the given parameters {distribution_params} do not match the expected parameters {expected_params}."
            )
        else:
            raise RuntimeError(
                f"The given distribution parameters {distribution_params} do not define the distribution type in the field 'type'. Please add a field 'type' and set it to one of the handeled types: {known_types}."
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sampler = get_risk_level_sampler(
        {"type": "log-uniform", "min": 0, "max": 1, "scale": 10}
    )
    # sampler = get_risk_level_sampler({"type": "normal", "mean": 0, "sigma": 1})
    a = sampler.sample(10000, "cpu").detach().numpy()
    _ = plt.hist(a, bins="auto")  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
