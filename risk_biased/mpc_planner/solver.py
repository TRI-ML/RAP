from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from mmcv import Config
import numpy as np
import torch

from risk_biased.mpc_planner.dynamics import PositionVelocityDoubleIntegrator
from risk_biased.mpc_planner.planner_cost import TrackingCost
from risk_biased.predictors.biased_predictor import LitTrajectoryPredictor
from risk_biased.utils.cost import BaseCostTorch
from risk_biased.utils.planner_utils import (
    AbstractState,
    to_state,
    evaluate_risk,
    get_interaction_cost,
)
from risk_biased.utils.risk import AbstractMonteCarloRiskEstimator


@dataclass
class CrossEntropySolverParams:
    """Dataclass for Cross Entropy Solver Parameters

    Args:
        num_control_samples: number of Monte Carlo samples for control input
        num_elite: number of elite samples
        iter_max: maximum iteration number
        smoothing_factor: smoothing factor in (0, 1) used to update the mean and the std of the
          control input distribution for the next iteration. If 0, the updated distribution is
          independent of the previous iteration. If 1, the updated distribution is the same as the
          previous iteration.
        mean_warm_start: internally saves control_input_mean of the last iteration of the current
          solve, so that control_input_mean will be warm-started in the next solve
    """

    num_control_samples: int
    num_elite: int
    iter_max: int
    smoothing_factor: float
    mean_warm_start: bool
    dt: float

    @staticmethod
    def from_config(cfg: Config):
        return CrossEntropySolverParams(
            cfg.num_control_samples,
            cfg.num_elite,
            cfg.iter_max,
            cfg.smoothing_factor,
            cfg.mean_warm_start,
            cfg.dt,
        )


class CrossEntropySolver:
    """Cross Entropy Solver for MPC Planner

    Args:
        params: CrossEntropySolverParams object
        dynamics_model: dynamics model for control
        control_input_mean: (num_agents, num_steps_future, control_dim) tensor of control input mean
        control_input_std: (num_agents, num_steps_future, control_dim) tensor of control input std
        tracking_cost_function: deterministic tracking cost that does not involve ado
        intraction_cost_function: interaction cost function between ego and (stochastic) ado
        risk_estimator (optional): Monte Carlo risk estimator for risk computation. If None,
          risk-neutral expecation is used for selectoin of elites. Defaults to None.
    """

    def __init__(
        self,
        params: CrossEntropySolverParams,
        dynamics_model: PositionVelocityDoubleIntegrator,
        control_input_mean: torch.Tensor,
        control_input_std: torch.Tensor,
        tracking_cost_function: TrackingCost,
        interaction_cost_function: BaseCostTorch,
        risk_estimator: Optional[AbstractMonteCarloRiskEstimator] = None,
    ) -> None:
        self.params = params

        self.control_input_mean_init = control_input_mean.detach().clone()
        self.control_input_std_init = control_input_std.detach().clone()
        assert (
            self.control_input_mean_init.shape == self.control_input_std_init.shape
        ), "control input mean and std must have the same size"
        assert (
            self.control_input_mean_init.shape[-1] == dynamics_model.control_dim
        ), f"control dimension must be {dynamics_model.control_dim}"

        self.dynamics_model = dynamics_model
        self.tracking_cost = tracking_cost_function
        self.interaction_cost = interaction_cost_function
        self.risk_estimator = risk_estimator

        self._iter_current = None
        self._control_input_mean = None
        self._control_input_std = None

        self._latest_ado_position_future_samples = None

        self.reset()

    def reset(self) -> None:
        """Resets the solver's internal state"""
        self._iter_current = 0
        self._control_input_mean = self.control_input_mean_init.clone()
        self._control_input_std = self.control_input_std_init.clone()
        self._latest_ado_position_future_samples = None

    def step(
        self,
        ego_state_history: AbstractState,
        ego_state_target_trajectory: AbstractState,
        ado_state_future_samples: AbstractState,
        weights: torch.Tensor,
        verbose: bool = False,
        risk_level: float = 0.0,
    ) -> Dict:
        """Performs one iteration step of the Cross Entropy Method

        Args:
            ego_state_history: (num_agents, num_steps)  ego state history
            ego_state_target_trajectory: (num_agents, num_steps_future) ego target
              state trajectory
            ado_state_future_samples: (num_prediction_samples, num_agents, num_steps_future)
                predicted ado trajectory samples
            weights: (num_prediction_samples, num_agents) prediction sample weight
            verbose (optional): Print progress. Defaults to False.
            risk_level (optional): a risk-level float for the solver. If 0.0, risk-neutral
              expectation is used for selection of elites. Defaults to 0.0.

        Return:
            Dictionary containing information about this solver step.
        """

        self._iter_current += 1
        ego_control_input = torch.normal(
            self._control_input_mean.expand(
                self.params.num_control_samples, -1, -1, -1
            ),
            self._control_input_std.expand(self.params.num_control_samples, -1, -1, -1),
        )
        if verbose:
            print(f"**Cross Entropy Iteration {self._iter_current}")
            print(
                f"****Drawring ego's control input samples of {ego_control_input.shape}"
            )
        ego_state_current = ego_state_history[..., -1]
        ego_state_future = self.dynamics_model.simulate(
            ego_state_current, ego_control_input
        )
        if verbose:
            print(f"****Simulating ego's future state trajectory")

        # state starts with x, y, angle, vx, vy
        tracking_cost = self.tracking_cost(
            ego_state_future.position,
            ego_state_target_trajectory.position,
            ego_state_target_trajectory.velocity,
        )
        if verbose:
            print(
                f"****Computing tracking cost of {tracking_cost.shape} for the control input samples"
            )

        # state starts with x, y
        interaction_cost = get_interaction_cost(
            ego_state_future,
            ado_state_future_samples,
            self.interaction_cost,
        )
        if verbose:
            print(
                f"****Computing interaction cost of {interaction_cost.shape} for the control input samples"
            )
        interaction_risk = evaluate_risk(
            risk_level,
            interaction_cost,
            weights=weights.permute(1, 0).unsqueeze(0).expand_as(interaction_cost),
            risk_estimator=self.risk_estimator,
        )

        total_risk = interaction_risk + tracking_cost
        elite_ego_control_input, elite_total_risk = self._get_elites(
            ego_control_input, total_risk
        )
        if verbose:
            print(f"****Selecting {self.params.num_elite} elite samples")
            print(f"****Elite Total_Risk Information: {elite_total_risk}")

        info = dict(
            iteration=self._iter_current,
            control_input_mean=self._control_input_mean.detach().cpu().numpy().copy(),
            control_input_std=self._control_input_std.detach().cpu().numpy().copy(),
            ego_state_future=ego_state_future.get_states(5)
            .detach()
            .cpu()
            .numpy()
            .copy(),
            ado_state_future_samples=ado_state_future_samples.get_states(5)
            .detach()
            .cpu()
            .numpy()
            .copy(),
            sample_weights=weights.detach().cpu().numpy().copy(),
            tracking_cost=tracking_cost.detach().cpu().numpy().copy(),
            interaction_cost=interaction_cost.detach().cpu().numpy().copy(),
            total_risk=total_risk.detach().cpu().numpy().copy(),
        )

        self._update_control_distribution(elite_ego_control_input)
        if verbose:
            print("****Updating ego's control distribution")

        return info

    def solve(
        self,
        predictor: LitTrajectoryPredictor,
        ego_state_history: AbstractState,
        ego_state_target_trajectory: AbstractState,
        ado_state_history: AbstractState,
        normalizer: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        num_prediction_samples: int = 1,
        verbose: bool = False,
        risk_level: float = 0.0,
        resample_prediction: bool = False,
        risk_in_predictor: bool = False,
    ) -> List[Dict]:
        """Performs Cross Entropy optimization of ego's control input

        Args:
            predictor: LitTrajectoryPredictor object
            ego_state_history: (num_agents, num_steps, state_dim) ego state history
            ego_state_target_trajectory: (num_agents, num_steps_future, state_dim) ego target
              state trajectory
            ado_state_history: (num_agents, num_steps, state_dim) ado state history
            normalizer: function that takes in an unnormalized trajectory and that outputs the
              normalized trajectory and the offset in this order
            num_prediction_samples: number of prediction samples. Defaults to 1.
            verbose (optional): Print progress. Defaults to False.
            risk_level (optional): a risk-level float for the entire prediction-planning pipeline.
              If 0.0, risk-neutral prediction and planning are used. Defaults to 0.0.
            resample_prediction (optional): If True, prediction is re-sampled in each cross-entropy
              iteration. Defaults to False.
            risk_in_predictor (optional): If True, risk-biased prediction is used and the solver
              becomes risk-neutral. If False, risk-neutral prediction is used and the solver becomes
              risk-sensitive. Defaults to False.

        Return:
            List of dictionaries each containing information about the corresponding solver step.
        """
        if risk_level == 0.0:
            risk_level_planner, risk_level_predictor = 0.0, 0.0
        else:
            if risk_in_predictor:
                risk_level_planner, risk_level_predictor = 0.0, risk_level
            else:
                risk_level_planner, risk_level_predictor = risk_level, 0.0
        self.reset()
        infos = []
        ego_state_future = self.dynamics_model.simulate(
            ego_state_history[..., -1],
            self.control_sequence,
        )
        for iter in range(self.params.iter_max):
            assert iter == self._iter_current
            if resample_prediction or self._iter_current == 0:
                ado_state_future_samples, weights = self.sample_prediction(
                    predictor,
                    ado_state_history,
                    normalizer,
                    ego_state_history,
                    ego_state_future,
                    num_prediction_samples,
                    risk_level_predictor,
                )
                self._latest_ado_position_future_samples = ado_state_future_samples
            info = self.step(
                ego_state_history,
                ego_state_target_trajectory,
                ado_state_future_samples,
                weights=weights,
                verbose=verbose,
                risk_level=risk_level_planner,
            )
            infos.append(info)
        if self.params.mean_warm_start:
            self.control_input_mean_init[:, :-1] = (
                self._control_input_mean[:, 1:].detach().clone()
            )
        return infos

    @property
    def control_sequence(self) -> torch.Tensor:
        """Returns the planned control sequence, which is a detached copy of the control input mean
        tensor

        Returns:
            (num_steps_future, control_dim) control sequence tensor
        """
        return self._control_input_mean.detach().clone()

    @staticmethod
    def sample_prediction(
        predictor: LitTrajectoryPredictor,
        ado_state_history: AbstractState,
        normalizer: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        ego_state_history: AbstractState,
        ego_state_future: AbstractState,
        num_prediction_samples: int = 1,
        risk_level: float = 0.0,
    ) -> Tuple[AbstractState, torch.Tensor]:
        """Sample prediction from the predictor given the history, normalizer, and the desired
        risk-level

        Args:
            predictor: LitTrajectoryPredictor object
            ado_state_history: (num_agents, num_steps, state_dim) tensor of ado position history
            normalizer: function that takes in an unnormalized trajectory and that outputs the
              normalized trajectory and the offset in this order
            ego_state_history: (num_agents, num_steps , state_dim) tensor of ego position history or future
            ego_state_future: (num_agents, num_steps_future, state_dim) tensor of ego position history or future
            num_prediction_samples (optional): number of prediction samples. Defaults to 1.
            risk_level (optional): a risk-level float for the predictor. If 0.0, risk-neutral
              prediction is sampled. Defaults to 0.0.

        Returns:
            state samples of shape (num_agents, num_prediction_samples, num_steps_future)
            probability weights of the samples of shape (num_agents, num_prediction_samples)
        """
        ado_position_history_normalized, offset = normalizer(
            ado_state_history.get_states(predictor.dynamic_state_dim)
            .unsqueeze(0)
            .expand(num_prediction_samples, -1, -1, -1)
        )

        x = ado_position_history_normalized.clone()
        mask_x = torch.ones_like(x[..., 0])
        map = torch.empty(num_prediction_samples, 0, 0, 2, device=x.device)
        mask_map = torch.empty(num_prediction_samples, 0, 0, device=x.device)

        batch = (
            x,
            mask_x,
            map,
            mask_map,
            offset,
            ego_state_history.get_states(predictor.dynamic_state_dim)
            .unsqueeze(0)
            .expand(num_prediction_samples, -1, -1, -1),
            ego_state_future.get_states(predictor.dynamic_state_dim)
            .unsqueeze(0)
            .expand(num_prediction_samples, -1, -1, -1),
        )

        ado_position_future_samples, weights = predictor.predict_step(
            batch,
            0,
            risk_level=risk_level,
            return_weights=True,
        )
        ado_position_future_samples = ado_position_future_samples.detach().cpu()
        weights = weights.detach().cpu()

        return to_state(ado_position_future_samples, predictor.dt), weights

    def fetch_latest_prediction(self):
        if self._latest_ado_position_future_samples is not None:
            return self._latest_ado_position_future_samples
        else:
            return None

    def _get_elites(
        self, control_input: torch.Tensor, risk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects elite control input based on corresponding risk (lower the better)

        Args:
            control_input: (num_control_samples, num_agents, num_steps_future, control_dim) control samples
            risk: (num_control_samples, num_agents) risk tensor

        Returns:
            elite_control_input: (num_elite, num_agents, num_steps_future, control_dim) elite control
            elite_risk: (num_elite, num_agents) elite risk
        """
        num_control_samples = self.params.num_control_samples
        assert (
            control_input.shape[0] == num_control_samples
        ), f"size of control_input tensor must be {num_control_samples} at dimension 0"
        assert (
            risk.shape[0] == num_control_samples
        ), f"size of risk tensor must be {num_control_samples} at dimension 0"

        _, sorted_risk_indices = torch.sort(risk, dim=0)
        elite_control_input = control_input[
            sorted_risk_indices[: self.params.num_elite], np.arange(risk.shape[1])
        ]
        elite_risk = risk[
            sorted_risk_indices[: self.params.num_elite], np.arange(risk.shape[1])
        ]
        return elite_control_input, elite_risk

    def _update_control_distribution(self, elite_control_input: torch.Tensor) -> None:
        """Updates control input distribution using elites

        Args:
            elite_control_input: (num_elite, num_steps_future, control_dim) elite control
        """
        num_elite, smoothing_factor = (
            self.params.num_elite,
            self.params.smoothing_factor,
        )
        assert (
            elite_control_input.shape[0] == num_elite
        ), f"size of elite_control_input tensor must be {num_elite} at dimension 0"

        elite_control_input_mean = elite_control_input.mean(dim=0, keepdim=False)
        if num_elite < 2:
            elite_control_input_std = torch.zeros_like(elite_control_input_mean)
        else:
            elite_control_input_std = elite_control_input.std(dim=0, keepdim=False)
        self._control_input_mean = (
            1.0 - smoothing_factor
        ) * elite_control_input_mean + smoothing_factor * self._control_input_mean
        self._control_input_std = (
            1.0 - smoothing_factor
        ) * elite_control_input_std + smoothing_factor * self._control_input_std
