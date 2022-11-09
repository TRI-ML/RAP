from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from mmcv import Config

from risk_biased.predictors.biased_predictor import LitTrajectoryPredictor
from risk_biased.mpc_planner.dynamics import PositionVelocityDoubleIntegrator
from risk_biased.mpc_planner.planner_cost import TrackingCostParams
from risk_biased.mpc_planner.solver import CrossEntropySolver, CrossEntropySolverParams
from risk_biased.mpc_planner.planner_cost import TrackingCost
from risk_biased.utils.cost import TTCCostTorch, TTCCostParams
from risk_biased.utils.planner_utils import AbstractState, to_state
from risk_biased.utils.risk import get_risk_estimator


@dataclass
class MPCPlannerParams:
    """Dataclass for MPC-Planner Parameters

    Args:
        dt_s: discrete time interval in seconds that is used for planning
        num_steps: number of time steps for which history of ego's and the other actor's
          trajectories are stored
        num_steps_future: number of time steps into the future for which ego's and the other actor's
          trajectories are considered
        acceleration_std_x_m_s2: Acceleration noise standard deviation (m/s^2) in x-direction that
          is used to initialize the Cross Entropy solver
        acceleration_std_y_m_s2: Acceleration noise standard deviation (m/s^2) in y-direction that
          is used to initialize the Cross Entropy solver
        risk_estimator_params: parameters for the Monte Carlo risk estimator used in the planner for
          ego's control optimization
        solver_params: parameters for the CrossEntropySolver
        tracking_cost_params: parameters for the TrackingCost
        ttc_cost_params: parameters for the TTCCost (i.e., collision cost between ego and the other
          actor)
    """

    dt: float
    num_steps: int
    num_steps_future: int
    acceleration_std_x_m_s2: float
    acceleration_std_y_m_s2: float

    risk_estimator_params: dict
    solver_params: CrossEntropySolverParams
    tracking_cost_params: TrackingCostParams
    ttc_cost_params: TTCCostParams

    @staticmethod
    def from_config(cfg: Config):
        return MPCPlannerParams(
            cfg.dt,
            cfg.num_steps,
            cfg.num_steps_future,
            cfg.acceleration_std_x_m_s2,
            cfg.acceleration_std_y_m_s2,
            cfg.risk_estimator,
            CrossEntropySolverParams.from_config(cfg),
            TrackingCostParams.from_config(cfg),
            TTCCostParams.from_config(cfg),
        )


class MPCPlanner:
    """MPC Planner with a Cross Entropy solver

    Args:
        params: MPCPlannerParams object
        predictor: LitTrajectoryPredictor object
        normalizer: function that takes in an unnormalized trajectory and that outputs the
          normalized trajectory and the offset in this order
    """

    def __init__(
        self,
        params: MPCPlannerParams,
        predictor: LitTrajectoryPredictor,
        normalizer: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:

        self.params = params
        self.dynamics_model = PositionVelocityDoubleIntegrator(params.dt)
        self.control_input_mean_init = torch.zeros(
            1, params.num_steps_future, self.dynamics_model.control_dim
        )
        self.control_input_std_init = torch.Tensor(
            [
                params.acceleration_std_x_m_s2,
                params.acceleration_std_y_m_s2,
            ]
        ).expand_as(self.control_input_mean_init)
        self.solver = CrossEntropySolver(
            params=params.solver_params,
            dynamics_model=self.dynamics_model,
            control_input_mean=self.control_input_mean_init,
            control_input_std=self.control_input_std_init,
            tracking_cost_function=TrackingCost(params.tracking_cost_params),
            interaction_cost_function=TTCCostTorch(params.ttc_cost_params),
            risk_estimator=get_risk_estimator(params.risk_estimator_params),
        )
        self.predictor = predictor
        self.normalizer = normalizer

        self._ego_state_history = []
        self._ego_state_target_trajectory = None
        self._ego_state_planned_trajectory = None

        self._ado_state_history = []
        self._latest_ado_position_future_samples = None

    def replan(
        self,
        current_ado_state: AbstractState,
        current_ego_state: AbstractState,
        target_velocity: torch.Tensor,
        num_prediction_samples: int = 1,
        risk_level: float = 0.0,
        resample_prediction: bool = False,
        risk_in_predictor: bool = False,
    ) -> None:
        """Performs re-planning given the current_ado_position, current_ego_state, and
        target_velocity. Updates ego_state_planned_trajectory. Note that all the information given
        to the solver.solve(...) is expressed in the ego-centric frame, whose origin is the initial
        ego position in ego_state_history and the x-direction is parallel to the initial ego
        velocity.

        Args:
            current_ado_position: ado state
            current_ego_state: ego state
            target_velocity: ((1), 2) tensor
            num_prediction_samples (optional): number of prediction samples. Defaults to 1.
            risk_level (optional): a risk-level float for the entire prediction-planning pipeline.
              If 0.0, risk-neutral prediction and planning are used. Defaults to 0.0.
            resample_prediction (optional): If True, prediction is re-sampled in each cross-entropy
              iteration. Defaults to False.
            risk_in_predictor (optional): If True, risk-biased prediction is used and the solver
              becomes risk-neutral. If False, risk-neutral prediction is used and the solver becomes
              risk-sensitive. Defaults to False.
        """
        self._update_ado_state_history(current_ado_state)
        self._update_ego_state_history(current_ego_state)
        self._update_ego_state_target_trajectory(current_ego_state, target_velocity)
        if not self.ado_state_history.shape[-1] < self.params.num_steps:
            self.solver.solve(
                self.predictor,
                self._map_to_ego_centric_frame(self.ego_state_history),
                self._map_to_ego_centric_frame(self._ego_state_target_trajectory),
                self._map_to_ego_centric_frame(self.ado_state_history),
                self.normalizer,
                num_prediction_samples=num_prediction_samples,
                risk_level=risk_level,
                resample_prediction=resample_prediction,
                risk_in_predictor=risk_in_predictor,
            )
        ego_state_planned_trajectory_in_ego_frame = self.dynamics_model.simulate(
            self._map_to_ego_centric_frame(self.ego_state_history[..., -1]),
            self.solver.control_sequence,
        )
        self._ego_state_planned_trajectory = self._map_to_world_frame(
            ego_state_planned_trajectory_in_ego_frame
        )
        latest_ado_position_future_samples_in_ego_frame = (
            self.solver.fetch_latest_prediction()
        )
        if latest_ado_position_future_samples_in_ego_frame is not None:
            self._latest_ado_position_future_samples = self._map_to_world_frame(
                latest_ado_position_future_samples_in_ego_frame
            )
        else:
            self._latest_ado_position_future_samples = None

    def get_planned_next_ego_state(self) -> AbstractState:
        """Returns the next ego state according to the ego_state_planned_trajectory

        Returns:
            Planned state
        """
        assert (
            self._ego_state_planned_trajectory is not None
        ), "call self.replan(...) first"
        return self._ego_state_planned_trajectory[..., 0]

    def reset(self) -> None:
        """Resets the planner's internal state. This will fully reset the solver's internal state,
        including solver.control_input_mean_init and solver.control_input_std_init."""
        self.solver.control_input_mean_init = (
            self.control_input_mean_init.detach().clone()
        )
        self.solver.control_input_std_init = (
            self.control_input_std_init.detach().clone()
        )
        self.solver.reset()

        self._ego_state_history = []
        self._ego_state_target_trajectory = None
        self._ego_state_planned_trajectory = None

        self._ado_state_history = []
        self._latest_ado_position_future_samples = None

    def fetch_latest_prediction(self) -> torch.Tensor:
        if self._latest_ado_position_future_samples is not None:
            return self._latest_ado_position_future_samples
        else:
            return None

    @property
    def ego_state_history(self) -> torch.Tensor:
        """Returns ego_state_history as a concatenated tensor
        Returns:
            ego_state_history tensor
        """
        assert len(self._ego_state_history) > 0
        return to_state(
            torch.stack(
                [ego_state.get_states(4) for ego_state in self._ego_state_history],
                dim=-2,
            ),
            self.params.dt,
        )

    @property
    def ado_state_history(self) -> torch.Tensor:
        """Returns ado_position_history as a concatenated tensor

        Returns:
            ado_position_history tensor
        """
        assert len(self._ado_state_history) > 0
        return to_state(
            torch.stack(
                [ado_state.get_states(4) for ado_state in self._ado_state_history],
                dim=-2,
            ),
            self.params.dt,
        )

    def _update_ego_state_history(self, current_ego_state: AbstractState) -> None:
        """Updates ego_state_history with the current_ego_state

        Args:
            current_ego_state: (1, state_dim) tensor
        """

        if len(self._ego_state_history) >= self.params.num_steps:
            self._ego_state_history = self._ego_state_history[1:]
        self._ego_state_history.append(current_ego_state)
        assert len(self._ego_state_history) <= self.params.num_steps

    def _update_ado_state_history(self, current_ado_state: AbstractState) -> None:
        """Updates ego_state_history with the current_ado_position

        Args:
            current_ado_state states of the current non-ego vehicles
        """

        if len(self._ado_state_history) >= self.params.num_steps:
            self._ado_state_history = self._ado_state_history[1:]
        self._ado_state_history.append(current_ado_state)
        assert len(self._ado_state_history) <= self.params.num_steps

    def _update_ego_state_target_trajectory(
        self, current_ego_state: AbstractState, target_velocity: torch.Tensor
    ) -> None:
        """Updates ego_state_target_trajectory based on the current_ego_state and the target_velocity

        Args:
            current_ego_state: state
            target_velocity: (1, 2) tensor
        """

        target_displacement = self.params.dt * target_velocity
        target_position_list = [current_ego_state.position]
        for time_idx in range(self.params.num_steps_future):
            target_position_list.append(target_position_list[-1] + target_displacement)
        target_position_list = target_position_list[1:]
        target_position = torch.cat(target_position_list, dim=-2)
        target_state = to_state(
            torch.cat(
                (target_position, target_velocity.expand_as(target_position)), dim=-1
            ),
            self.params.dt,
        )
        self._ego_state_target_trajectory = target_state

    def _map_to_ego_centric_frame(
        self, trajectory_in_world_frame: AbstractState
    ) -> torch.Tensor:
        """Maps trajectory epxressed in the world frame to the ego-centric frame, whose origin is
        the initial ego position in ego_state_history and the x-direction is parallel to the initial
        ego velocity

        Args:
            trajectory: sequence of states

        Returns:
            trajectory mapped to the ego-centric frame
        """
        # If trajectory_in_world_frame is of shape (..., state_dim) then use the associated
        # dynamics model in translate_position and rotate_angle. Otherwise assume that th
        # trajectory is in the 2D position space.

        ego_pos_init = self.ego_state_history.position[..., -1, :]
        ego_vel_init = self.ego_state_history.velocity[..., -1, :]
        ego_rot_init = torch.atan2(ego_vel_init[..., 1], ego_vel_init[..., 0])
        trajectory_in_ego_frame = trajectory_in_world_frame.translate(
            -ego_pos_init
        ).rotate(-ego_rot_init)
        return trajectory_in_ego_frame

    def _map_to_world_frame(
        self, trajectory_in_ego_frame: torch.Tensor
    ) -> torch.Tensor:
        """Maps trajectory epxressed in the ego-centric frame to the world frame

        Args:
            trajectory_in_ego_frame: (..., 2) position trajectory or (..., markov_state_dim) state
              trajectory expressed in the ego-centric frame, whose origin is the initial ego
              position in ego_state_history and the x-direction is parallel to the initial ego
              velocity

        Returns:
            trajectory mapped to the world frame
        """
        # state starts with x, y, angle
        ego_pos_init = self.ego_state_history.position[..., -1, :]
        ego_rot_init = self.ego_state_history.angle[..., -1, :]
        trajectory_in_world_frame = trajectory_in_ego_frame.rotate(
            ego_rot_init
        ).translate(ego_pos_init)
        return trajectory_in_world_frame
