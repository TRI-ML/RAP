from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple, Union


from einops import repeat
from mmcv import Config
import pytorch_lightning as pl
import torch

from risk_biased.models.cvae_params import CVAEParams
from risk_biased.models.biased_cvae_model import (
    cvae_factory,
)

from risk_biased.utils.cost import TTCCostTorch, TTCCostParams
from risk_biased.utils.risk import get_risk_estimator
from risk_biased.utils.risk import get_risk_level_sampler


@dataclass
class LitTrajectoryPredictorParams:
    """
    cvae_params: CVAEParams class defining the necessary parameters for the CVAE model
    risk distribution: dict of string and values defining the risk distribution to use
    risk_estimator: dict of string and values defining the risk estimator to use
    kl_weight: float defining the weight of the KL term in the loss function
    kl_threshold: float defining the threshold to apply when computing kl divergence (avoid posterior collapse)
    risk_weight: float defining the weight of the risk term in the loss function
    n_mc_samples_risk: int defining the number of Monte Carlo samples to use when estimating the risk
    n_mc_samples_biased: int defining the number of Monte Carlo samples to use when estimating the expected biased cost
    dt: float defining the duration between two consecutive time steps
    learning_rate: float defining the learning rate for the optimizer
    use_risk_constraint: bool defining whether to use the risk constrained optimization procedure
    risk_constraint_update_every_n_epoch: int defining the number of epochs between two risk weight updates
    risk_constraint_weight_update_factor: float defining the factor by which the risk weight is multiplied at each update
    risk_constraint_weight_maximum: float defining the maximum value of the risk weight
    num_samples_min_fde: int defining the number of samples to use when estimating the minimum FDE
    condition_on_ego_future: bool defining whether to condition the biasing on the ego future trajectory (else on the ego past)

    """

    cvae_params: CVAEParams
    risk_distribution: dict
    risk_estimator: dict
    kl_weight: float
    kl_threshold: float
    risk_weight: float
    n_mc_samples_risk: int
    n_mc_samples_biased: int
    dt: float
    learning_rate: float
    use_risk_constraint: bool
    risk_constraint_update_every_n_epoch: int
    risk_constraint_weight_update_factor: float
    risk_constraint_weight_maximum: float
    num_samples_min_fde: int
    condition_on_ego_future: bool

    @staticmethod
    def from_config(cfg: Config):
        cvae_params = CVAEParams.from_config(cfg)
        return LitTrajectoryPredictorParams(
            risk_distribution=cfg.risk_distribution,
            risk_estimator=cfg.risk_estimator,
            kl_weight=cfg.kl_weight,
            kl_threshold=cfg.kl_threshold,
            risk_weight=cfg.risk_weight,
            n_mc_samples_risk=cfg.n_mc_samples_risk,
            n_mc_samples_biased=cfg.n_mc_samples_biased,
            dt=cfg.dt,
            learning_rate=cfg.learning_rate,
            cvae_params=cvae_params,
            use_risk_constraint=cfg.use_risk_constraint,
            risk_constraint_update_every_n_epoch=cfg.risk_constraint_update_every_n_epoch,
            risk_constraint_weight_update_factor=cfg.risk_constraint_weight_update_factor,
            risk_constraint_weight_maximum=cfg.risk_constraint_weight_maximum,
            num_samples_min_fde=cfg.num_samples_min_fde,
            condition_on_ego_future=cfg.condition_on_ego_future,
        )


class LitTrajectoryPredictor(pl.LightningModule):
    """Pytorch Lightning Module for Trajectory Prediction with the biased cvae model

    Args:
        params : dataclass object containing the necessary parameters
        cost_params: dataclass object defining the TTC cost function
        unnormalizer: function that takes in a trajectory and an offset and that outputs the
                      unnormalized trajectory
    """

    def __init__(
        self,
        params: LitTrajectoryPredictorParams,
        cost_params: TTCCostParams,
        unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        model = cvae_factory(
            params.cvae_params,
            cost_function=TTCCostTorch(cost_params),
            risk_estimator=get_risk_estimator(params.risk_estimator),
            training_mode="cvae",
        )
        self.model = model
        self.params = params
        self._unnormalize_trajectory = unnormalizer
        self.set_training_mode("cvae")

        self.learning_rate = params.learning_rate
        self.num_samples_min_fde = params.num_samples_min_fde

        self.dynamic_state_dim = params.cvae_params.dynamic_state_dim
        self.dt = params.cvae_params.dt

        self.use_risk_constraint = params.use_risk_constraint
        self.risk_weight = params.risk_weight
        self.risk_weight_ratio = params.risk_weight / params.kl_weight
        self.kl_weight = params.kl_weight
        if self.use_risk_constraint:
            self.risk_constraint_update_every_n_epoch = (
                params.risk_constraint_update_every_n_epoch
            )
            self.risk_constraint_weight_update_factor = (
                params.risk_constraint_weight_update_factor
            )
            self.risk_constraint_weight_maximum = params.risk_constraint_weight_maximum

        self._risk_sampler = get_risk_level_sampler(params.risk_distribution)

    def set_training_mode(self, training_mode: str):
        self.model.set_training_mode(training_mode)
        self.partial_get_loss = partial(
            self.model.get_loss,
            kl_threshold=self.params.kl_threshold,
            n_samples_risk=self.params.n_mc_samples_risk,
            n_samples_biased=self.params.n_mc_samples_biased,
            dt=self.params.dt,
            unnormalizer=self._unnormalize_trajectory,
        )

    def _get_loss(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        map: torch.Tensor,
        mask_map: torch.Tensor,
        y: torch.Tensor,
        mask_y: torch.Tensor,
        mask_loss: torch.Tensor,
        x_ego: torch.Tensor,
        y_ego: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
        risk_level: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], dict]:
        """Compute loss based on trajectory history x and future y

        Args:
            x: (batch_size, num_agents, num_steps, state_dim) tensor of history
            mask_x: (batch_size, num_agents, num_steps) tensor of bool mask
            map: (batch_size, num_objects, object_sequence_length, map_feature_dim) tensor of encoded map objects
            mask_map: (batch_size, num_objects, object_sequence_length) tensor True where map features are good False where it is padding
            y: (batch_size, num_agents, num_steps_future, state_dim) tensor of future trajectory.
            mask_y: (batch_size, num_agents, num_steps_future) tensor of bool mask.
            mask_loss: (batch_size, num_agents, num_steps_future) tensor of bool mask set to True where the loss
            should be computed and to False where it shouldn't
            offset : (batch_size, num_agents, state_dim) offset position from ego
            risk_level : (batch_size, num_agents) tensor of risk levels desired for future trajectories

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]: (1,) loss tensor or tuple of
            loss tensors
            dict: dict that contains values to be logged
        """
        return self.partial_get_loss(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            y=y,
            mask_y=mask_y,
            mask_loss=mask_loss,
            offset=offset,
            risk_level=risk_level,
            x_ego=x_ego,
            y_ego=y_ego,
            risk_weight=self.risk_weight,
            kl_weight=self.kl_weight,
        )

    def log_with_prefix(
        self,
        log_dict: dict,
        prefix: Optional[str] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
    ) -> None:
        """log entries in log_dict while optinally adding "<prefix>/" to its keys

        Args:
            log_dict: dict that contains values to be logged
            prefix: prefix to be added to keys
            on_step: if True logs at this step. None auto-logs at the training_step but not
            validation/test_step
            on_epoch: if True logs epoch accumulated metrics. None auto-logs at the val/test
            step but not training_step
        """
        if prefix is None:
            prefix = ""
        else:
            prefix += "/"

        for (metric, value) in log_dict.items():
            metric = prefix + metric
            self.log(metric, value, on_step=on_step, on_epoch=on_epoch)

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]:
        """Configure optimizer for PyTorch-Lightning

        Returns:
            torch.optim.Optimizer: optimizer to be used for training
        """
        if isinstance(self.model.get_parameters(), list):
            self._optimizers = [
                torch.optim.Adam(params, lr=self.learning_rate)
                for params in self.model.get_parameters()
            ]
        else:
            self._optimizers = [
                torch.optim.Adam(self.model.get_parameters(), lr=self.learning_rate)
            ]
        return self._optimizers

    def training_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
    ) -> dict:
        """Training step definition for PyTorch-Lightning

        Args:
            batch : [(batch_size, num_agents, num_steps, state_dim),           # past trajectories of all agents in the scene
                     (batch_size, num_agents, num_steps),                      # mask past False where past trajectories are padding data
                     (batch_size, num_agents, num_steps_future, state_dim),    # future trajectory
                     (batch_size, num_agents, num_steps_future),               # mask future False where future trajectories are padding data
                     (batch_size, num_agents, num_steps_future),               # mask loss False where future trajectories are not to be predicted
                     (batch_size, num_objects, object_seq_len, state_dim),     # map object sequences in the scene
                     (batch_size, num_objects, object_seq_len),                # mask map False where map objects are padding data
                     (batch_size, num_agents, state_dim),                      # position offset of all agents relative to ego at present time
                     (batch_size, 1, num_steps, state_dim),                    # ego past trajectory
                     (batch_size, 1, num_steps_future, state_dim)]             # ego future trajectory
            batch_idx : batch_idx to be used by PyTorch-Lightning

        Returns:
            dict: dict of outputs containing loss
        """
        x, mask_x, y, mask_y, mask_loss, map, mask_map, offset, x_ego, y_ego = batch
        risk_level = repeat(
            self._risk_sampler.sample(x.shape[0], x.device),
            "b -> b num_agents",
            num_agents=x.shape[1],
        )
        loss, log_dict = self._get_loss(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            y=y,
            mask_y=mask_y,
            mask_loss=mask_loss,
            offset=offset,
            risk_level=risk_level,
            x_ego=x_ego,
            y_ego=y_ego,
        )
        if isinstance(loss, tuple):
            loss = sum(loss)
        self.log_with_prefix(log_dict, prefix="train", on_step=True, on_epoch=False)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[dict]) -> None:
        """Called at the end of the training epoch with the outputs of all training steps

        Args:
            outputs: list of outputs of all training steps in the current epoch
        """
        if self.use_risk_constraint:
            if (
                self.model.training_mode == "bias"
                and (self.trainer.current_epoch + 1)
                % self.risk_constraint_update_every_n_epoch
                == 0
            ):
                self.risk_weight_ratio *= self.risk_constraint_weight_update_factor
                if self.risk_weight_ratio < self.risk_constraint_weight_maximum:
                    sum_weight = self.risk_weight + self.kl_weight
                    self.risk_weight = (
                        sum_weight
                        * self.risk_weight_ratio
                        / (1 + self.risk_weight_ratio)
                    )
                    self.kl_weight = sum_weight / (1 + self.risk_weight_ratio)
                # self.risk_weight *= self.risk_constraint_weight_update_factor
                # if self.risk_weight > self.risk_constraint_weight_maximum:
                #     self.risk_weight = self.risk_constraint_weight_maximum

    def _get_risk_tensor(
        self,
        batch_size: int,
        num_agents: int,
        device: torch.device,
        risk_level: Optional[torch.Tensor] = None,
    ):
        """This function is used to reformat different possible formattings of risk_level input arguments into a tensor of shape (batch_size).
            If given a tensor the same tensor is returned.
            If given a float value, a tensor of this value is returned.
            If given None, a tensor filled with random samples is returned.

        Args:
            batch_size : desired batch size
            device : device on which we want to store risk
            risk_level : The risk level as a tensor, a float value or None

        Returns:
            _type_: _description_
        """
        if risk_level is not None:
            if isinstance(risk_level, (float, int)):
                risk_level = (
                    torch.ones(batch_size, num_agents, device=device) * risk_level
                )
            else:
                risk_level = risk_level.to(device)
        else:
            risk_level = None

        return risk_level

    def validation_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
        risk_level: float = 1.0,
    ) -> dict:
        """Validation step definition for PyTorch-Lightning

        Args:
            batch : [(batch_size, num_agents, num_steps, state_dim),           # past trajectories of all agents in the scene
                     (batch_size, num_agents, num_steps),                      # mask past False where past trajectories are padding data
                     (batch_size, num_agents, num_steps_future, state_dim),    # future trajectory
                     (batch_size, num_agents, num_steps_future),               # mask future False where future trajectories are padding data
                     (batch_size, num_agents, num_steps_future),               # mask loss False where future trajectories are not to be predicted
                     (batch_size, num_objects, object_seq_len, state_dim),     # map object sequences in the scene
                     (batch_size, num_objects, object_seq_len),                # mask map False where map objects are padding data
                     (batch_size, num_agents, state_dim),                      # position offset of all agents relative to ego at present time
                     (batch_size, 1, num_steps, state_dim),                    # ego past trajectory
                     (batch_size, 1, num_steps_future, state_dim)]             # ego future trajectory
            batch_idx : batch_idx to be used by PyTorch-Lightning
            risk_level : optional desired risk level

        Returns:
            dict: dict of outputs containing loss
        """
        x, mask_x, y, mask_y, mask_loss, map, mask_map, offset, x_ego, y_ego = batch

        risk_level = self._get_risk_tensor(
            x.shape[0], x.shape[1], x.device, risk_level=risk_level
        )
        self.model.eval()
        log_dict_accuracy = self.model.get_prediction_accuracy(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            y=y,
            mask_loss=mask_loss,
            offset=offset,
            x_ego=x_ego,
            y_ego=y_ego,
            unnormalizer=self._unnormalize_trajectory,
            risk_level=risk_level,
            num_samples_min_fde=self.num_samples_min_fde,
        )

        loss, log_dict_loss = self._get_loss(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            y=y,
            mask_y=mask_y,
            mask_loss=mask_loss,
            offset=offset,
            risk_level=risk_level,
            x_ego=x_ego,
            y_ego=y_ego,
        )

        if isinstance(loss, tuple):
            loss = sum(loss)

        self.log_with_prefix(
            dict(log_dict_accuracy, **log_dict_loss),
            prefix="val",
            on_step=False,
            on_epoch=True,
        )
        self.model.train()
        return {"loss": loss}

    def test_step(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
        risk_level: Optional[torch.Tensor] = None,
    ) -> dict:
        """Test step definition for PyTorch-Lightning

        Args:
            batch : [(batch_size, num_agents, num_steps, state_dim),           # past trajectories of all agents in the scene
                     (batch_size, num_agents, num_steps),                      # mask past False where past trajectories are padding data
                     (batch_size, num_agents, num_steps_future, state_dim),    # future trajectory
                     (batch_size, num_agents, num_steps_future),               # mask future False where future trajectories are padding data
                     (batch_size, num_agents, num_steps_future),               # mask loss False where future trajectories are not to be predicted
                     (batch_size, num_objects, object_seq_len, state_dim),     # map object sequences in the scene
                     (batch_size, num_objects, object_seq_len),                # mask map False where map objects are padding data
                     (batch_size, num_agents, state_dim),                      # position offset of all agents relative to ego at present time
                     (batch_size, 1, num_steps, state_dim),                    # ego past trajectory
                     (batch_size, 1, num_steps_future, state_dim)]             # ego future trajectory
            batch_idx : batch_idx to be used by PyTorch-Lightning
            risk_level : optional desired risk level

        Returns:
            dict: dict of outputs containing loss
        """
        x, mask_x, y, mask_y, mask_loss, map, mask_map, offset, x_ego, y_ego = batch
        risk_level = self._get_risk_tensor(
            x.shape[0], x.shape[1], x.device, risk_level=risk_level
        )
        loss, log_dict = self._get_loss(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            y=y,
            mask_y=mask_y,
            mask_loss=mask_loss,
            offset=offset,
            risk_level=risk_level,
            x_ego=x_ego,
            y_ego=y_ego,
        )
        if isinstance(loss, tuple):
            loss = sum(loss)
        self.log_with_prefix(log_dict, prefix="test", on_step=False, on_epoch=True)
        return {"loss": loss}

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int = 0,
        risk_level: Optional[torch.Tensor] = None,
        n_samples: int = 0,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict step definition for PyTorch-Lightning

        Args:
            batch: [(batch_size, num_agents, num_steps, state_dim),           # past trajectories of all agents in the scene
                    (batch_size, num_agents, num_steps),                      # mask past False where past trajectories are padding data
                    (batch_size, num_objects, object_seq_len, state_dim),     # map object sequences in the scene
                    (batch_size, num_objects, object_seq_len),                # mask map False where map objects are padding data
                    (batch_size, num_agents, state_dim),                      # position offset of all agents relative to ego at present time
                    (batch_size, 1, num_steps, state_dim),                    # past trajectory of the ego agent in the scene
                    (batch_size, 1, num_steps_future, state_dim),]            # future trajectory of the ego agent in the scene
            batch_idx : batch_idx to be used by PyTorch-Lightning (unused here)
            risk_level : optional desired risk level
            n_samples: Number of samples to predict per agent
                With value of 0 does not include the `n_samples` dim in the output.
            return_weights: If True, also returns the sample weights

        Returns:
            (batch_size, num_agents, (n_samples), num_steps_future, state_dim) tensor (and if run with return_weights=True, also a (batch_size, num_agents) tensor)
        """
        x, mask_x, map, mask_map, offset, x_ego, y_ego = batch
        risk_level = self._get_risk_tensor(
            batch_size=x.shape[0],
            num_agents=x.shape[1],
            device=x.device,
            risk_level=risk_level,
        )
        y_sampled, weights, _ = self.model(
            x,
            mask_x,
            map,
            mask_map,
            offset=offset,
            x_ego=x_ego,
            y_ego=y_ego,
            risk_level=risk_level,
            n_samples=n_samples,
        )
        predict_sampled = self._unnormalize_trajectory(y_sampled, offset)
        if return_weights:
            return predict_sampled, weights
        else:
            return predict_sampled

    def predict_loop_once(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int = 0,
        risk_level: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Predict with refinment:
                A first prediction is done as in predict_step, however instead of unnormalize and return it,
                it is fed to the encoder that wast trained to encode past and ground truth future.
                Then the decoder is used again but its latent input sample is biased by the encoder
                instead of being a sample of the prior distribution.
                Then as in predict_step the result is unnormalized and returned.

        Args:
            batch: [(batch_size, num_agents, num_steps, state_dim),           # past trajectories of all agents in the scene
                    (batch_size, num_agents, num_steps),                      # mask past False where past trajectories are padding data
                    (batch_size, num_objects, object_seq_len, state_dim),     # map object sequences in the scene
                    (batch_size, num_objects, object_seq_len),                # mask map False where map objects are padding data
                    (batch_size, num_agents, state_dim),]                     # position offset of all agents relative to ego at present time
            batch_idx : batch_idx to be used by PyTorch-Lightning (Unused here). Defaults to 0.
            risk_level : optional desired risk level
            return_weights: If True, also returns the sample weights

        Returns:
            torch.Tensor: (batch_size, num_steps_future, state_dim) tensor
        """
        x, mask_x, map, mask_map, offset = batch
        risk_level = self._get_risk_tensor(
            x.shape[0], x.shape[1], x.device, risk_level=risk_level
        )
        y_sampled, weights, _ = self.model(
            x,
            mask_x,
            map,
            mask_map,
            offset=offset,
            risk_level=risk_level,
        )
        mask_y = repeat(mask_x.any(-1), "b a -> b a f", f=y_sampled.shape[-2])
        y_sampled, weights, _ = self.model(
            x,
            mask_x,
            map,
            mask_map,
            y_sampled,
            mask_y,
            offset=offset,
            risk_level=risk_level,
        )
        predict_sampled = self._unnormalize_trajectory(y_sampled, offset=offset)
        if return_weights:
            return predict_sampled, weights
        else:
            return predict_sampled
