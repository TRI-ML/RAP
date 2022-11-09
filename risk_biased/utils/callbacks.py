import copy
from dataclasses import dataclass

from mmcv import Config
import matplotlib.pyplot as plt
import numpy as np
from pydantic import NoneBytes
import pytorch_lightning as pl
import torch
import wandb

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.cost import (
    DistanceCostNumpy,
    DistanceCostParams,
    TTCCostNumpy,
    TTCCostParams,
)
from risk_biased.utils.risk import get_risk_level_sampler


class SwitchTrainingModeCallback(pl.Callback):
    """
    This callback switches between CVAE traning and biasing training for the biased_latent_cvae_model
    Args:
        switch_at_epoch: The number of epoch after which to make the switch. The CVAE is not trained anymore after that point.
    """

    def __init__(self, switch_at_epoch: int) -> None:
        super().__init__()
        self._switch_at_epoch = switch_at_epoch
        self._train_has_started = False

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Store the optimizer list and set the trainer to the first optimizer."""
        self._optimizers = trainer.optimizers
        trainer.optimizers = [self._optimizers[0]]
        self._train_has_started = True

    def on_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Check if the switch should be made and if so,
        set the trainer on the second optimizer.
        """
        if trainer.current_epoch == self._switch_at_epoch and self._train_has_started:
            print("Switching to bias training.")
            pl_module.set_training_mode("bias")
            trainer.optimizers = [self._optimizers[1]]


def get_fast_slow_scenes(params: RandomSceneParams, n_samples: int):
    """Define and return two RandomScene objects, one initialized such that slow
       pedestrians are safer and the other such that fast pedestrians are safer.

    Args:
        params: dataclass containing the necessary parameters for a RandomScene object
        n_samples: number of samples to draw in each scene
    """
    params = copy.deepcopy(params)
    params.batch_size = n_samples
    scene_safe_slow = RandomScene(
        params,
        is_torch=False,
    )
    percent_right = 0.8
    percent_top = 1.1
    angle = 5 * np.pi / 4
    positions = np.array([[[percent_right, percent_top]]] * n_samples)
    angles = np.array([[angle]] * n_samples)
    scene_safe_slow.set_pedestrians_states(positions, angles)

    scene_safe_fast = RandomScene(
        params,
        is_torch=False,
    )
    percent_right = 0.8
    percent_top = 0.6
    angle = 5 * np.pi / 4
    positions = np.array([[[percent_right, percent_top]]] * n_samples)
    angles = np.array([[angle]] * n_samples)
    scene_safe_fast.set_pedestrians_states(positions, angles)
    return scene_safe_fast, scene_safe_slow


@dataclass
class DrawCallbackParams:
    """
    Args:
        scene_params: dataclass parameters for the RandomScene
        dist_cost_params: dataclass parameters for the DistanceCost
        ttc_cost_params: dataclass parameters for the TTCCost
        plot_interval_epoch: number of epochs between each plot drawing
        histogram_interval_epoch: number of epochs between each histogram drawing
        num_steps: number of time steps as defined in the config
        num_steps_future: number of time steps in the future as defined in the config
        risk_distribution: dict object describing a risk distribution
        dt: time step size as defined in the config
    """

    scene_params: RandomSceneParams
    dist_cost_params: DistanceCostParams
    ttc_cost_params: TTCCostParams
    plot_interval_epoch: int
    histogram_interval_epoch: int
    num_steps: int
    num_steps_future: int
    risk_distribution: dict
    dt: float

    @staticmethod
    def from_config(cfg: Config):
        return DrawCallbackParams(
            scene_params=RandomSceneParams.from_config(cfg),
            dist_cost_params=DistanceCostParams.from_config(cfg),
            ttc_cost_params=TTCCostParams.from_config(cfg),
            plot_interval_epoch=cfg.plot_interval_epoch,
            histogram_interval_epoch=cfg.histogram_interval_epoch,
            num_steps=cfg.num_steps,
            num_steps_future=cfg.num_steps_future,
            risk_distribution=cfg.risk_distribution,
            dt=cfg.dt,
        )


class HistogramCallback(pl.Callback):
    """Logs histograms of distances, distance cost and ttc cost for the data, the predictions at risk_level=0, the predictions at risk_level=1
    Args:
        params: dataclass defining the necessary parameters
        n_samples: Number of samples to use for the histogram plot
    """

    def __init__(
        self,
        params: DrawCallbackParams,
        n_samples=1000,
    ):
        super().__init__()
        self.scene_safe_fast, self.scene_safe_slow = get_fast_slow_scenes(
            params.scene_params, n_samples
        )
        self.num_steps = params.num_steps
        self.n_scenes = n_samples
        self.sample_times = params.scene_params.sample_times
        self.dist_cost_func = DistanceCostNumpy(params.dist_cost_params)
        self.ttc_cost_func = TTCCostNumpy(params.ttc_cost_params)
        self.histogram_interval_epoch = params.histogram_interval_epoch

        self.ego_traj = self.scene_safe_fast.get_ego_ref_trajectory(self.sample_times)

        self._risk_sampler = get_risk_level_sampler(params.risk_distribution)

    def _log_scene(self, pl_module: pl.LightningModule, scene: RandomScene, name: str):
        """
        Log in WandB three histogram for the given scene: One for the data, one for the predictions at risk_level=0 and one for the predictions at risk_level=1
        Args:
            pl_module: LightningModule object
            scene: RandomScene object
            name: name of the given scene
        """
        ped_trajs = scene.get_pedestrians_trajectories()
        device = pl_module.device
        n_agents = ped_trajs.shape[1]

        input_traj = ped_trajs[..., : self.num_steps, :]

        normalized_input, offset = SceneDataLoaders.normalize_trajectory(
            torch.from_numpy(input_traj.astype("float32")).contiguous().to(device)
        )
        mask_input = torch.ones_like(normalized_input[..., 0])
        ego_history = (
            torch.from_numpy(self.ego_traj[..., : self.num_steps, :].astype("float32"))
            .expand_as(normalized_input)
            .contiguous()
            .to(device)
        )
        ego_future = (
            torch.from_numpy(self.ego_traj[..., self.num_steps :, :].astype("float32"))
            .expand(normalized_input.shape[0], n_agents, -1, -1)
            .contiguous()
            .to(device)
        )
        map = torch.empty(ego_history.shape[0], 0, 0, 2, device=mask_input.device)
        mask_map = torch.empty(ego_history.shape[0], 0, 0, device=mask_input.device)

        pred_riskier = (
            pl_module.predict_step(
                (
                    normalized_input,
                    mask_input,
                    map,
                    mask_map,
                    offset,
                    ego_history,
                    ego_future,
                ),
                0,
                risk_level=self._risk_sampler.get_highest_risk(
                    batch_size=self.n_scenes, device=device
                )
                .unsqueeze(1)
                .repeat(1, n_agents),
            )
            .cpu()
            .detach()
            .numpy()
        )

        pred = (
            pl_module.predict_step(
                (
                    normalized_input,
                    mask_input,
                    map,
                    mask_map,
                    offset,
                    ego_history,
                    ego_future,
                ),
                0,
                risk_level=None,
            )
            .cpu()
            .detach()
            .numpy()
        )

        ped_trajs_pred = np.concatenate((input_traj, pred), axis=-2)
        ped_trajs_pred_riskier = np.concatenate((input_traj, pred_riskier), axis=-2)

        travel_distances = np.sqrt(
            np.square(ped_trajs[..., -1, :] - ped_trajs[..., 0, :]).sum(-1)
        )

        dist_cost, dist = self.dist_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs[..., self.num_steps :, :],
        )

        ttc_cost, (ttc, dist) = self.ttc_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs[..., self.num_steps :, :],
            scene.get_ego_ref_velocity(),
            scene.get_pedestrians_velocities(),
        )

        travel_distances_pred = np.sqrt(
            np.square(ped_trajs_pred[..., -1, :] - ped_trajs_pred[..., 0, :]).sum(-1)
        )
        dist_cost_pred, dist_pred = self.dist_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs_pred[..., self.num_steps :, :],
        )
        sample_times = np.array(self.sample_times)
        ped_velocities_pred = (
            ped_trajs_pred[..., 1:, :] - ped_trajs_pred[..., :-1, :]
        ) / ((sample_times[1:] - sample_times[:-1])[None, None, :, None])
        ped_velocities_pred = np.concatenate(
            (ped_velocities_pred[..., 0:1, :], ped_velocities_pred), -2
        )
        ttc_cost_pred, (ttc_pred, dist_pred) = self.ttc_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs_pred[..., self.num_steps :, :],
            scene.get_ego_ref_velocity(),
            ped_velocities_pred[..., self.num_steps :, :],
        )

        travel_distances_pred_riskier = np.sqrt(
            np.square(
                ped_trajs_pred_riskier[..., -1, :] - ped_trajs_pred_riskier[..., 0, :]
            ).sum(-1)
        )

        dist_cost_pred_riskier, dist_pred_riskier = self.dist_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs_pred_riskier[..., self.num_steps :, :],
        )
        sample_times = np.array(self.sample_times)
        ped_velocities_pred_riskier = (
            ped_trajs_pred_riskier[..., 1:, :] - ped_trajs_pred_riskier[..., :-1, :]
        ) / ((sample_times[1:] - sample_times[:-1])[None, None, :, None])
        ped_velocities_pred_riskier = np.concatenate(
            (ped_velocities_pred_riskier[..., 0:1, :], ped_velocities_pred_riskier), -2
        )
        ttc_cost_pred_riskier, (ttc_pred, dist_pred_riskier) = self.ttc_cost_func(
            self.ego_traj[..., self.num_steps :, :],
            ped_trajs_pred_riskier[..., self.num_steps :, :],
            scene.get_ego_ref_velocity(),
            ped_velocities_pred_riskier[..., self.num_steps :, :],
        )
        data = [
            [dist, dist_pred, dist_risk]
            for (dist, dist_pred, dist_risk) in zip(
                travel_distances.flatten(),
                travel_distances_pred.flatten(),
                travel_distances_pred_riskier.flatten(),
            )
        ]
        table_travel_distance = wandb.Table(
            data=data,
            columns=[
                "Travel distance data " + name,
                "Travel distance prediction " + name,
                "Travel distance riskier " + name,
            ],
        )
        data = [
            [cost, cost_pred, cost_risk]
            for (cost, cost_pred, cost_risk) in zip(
                dist_cost.flatten(),
                dist_cost_pred.flatten(),
                dist_cost_pred_riskier.flatten(),
            )
        ]
        table_distance_cost = wandb.Table(
            data=data,
            columns=[
                "Distance cost data " + name,
                "Distance cost prediction " + name,
                "Distance cost riskier " + name,
            ],
        )
        data = [
            [ttc, ttc_pred, ttc_risk]
            for (ttc, ttc_pred, ttc_risk) in zip(
                ttc_cost.flatten(),
                ttc_cost_pred.flatten(),
                ttc_cost_pred_riskier.flatten(),
            )
        ]
        table_ttc_cost = wandb.Table(
            data=data,
            columns=[
                "TTC cost data " + name,
                "TTC cost prediction " + name,
                "TTC cost riskier " + name,
            ],
        )
        wandb.log(
            {
                "Travel distance data "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_01_bins",
                    data_table=table_travel_distance,
                    fields={
                        "value": "Travel distance data " + name,
                        "title": "Travel distance data " + name,
                    },
                ),
                "Travel distance prediction "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_01_bins",
                    data_table=table_travel_distance,
                    fields={
                        "value": "Travel distance prediction " + name,
                        "title": "Travel distance prediction " + name,
                    },
                ),
                "Travel distance riskier "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_01_bins",
                    data_table=table_travel_distance,
                    fields={
                        "value": "Travel distance riskier " + name,
                        "title": "Travel distance riskier " + name,
                    },
                ),
                "Distance cost data "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_0025_bins",
                    data_table=table_distance_cost,
                    fields={
                        "value": "Distance cost data " + name,
                        "title": "Distance cost data " + name,
                    },
                ),
                "Distance cost prediction "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_0025_bins",
                    data_table=table_distance_cost,
                    fields={
                        "value": "Distance cost prediction " + name,
                        "title": "Distance cost prediction " + name,
                    },
                ),
                "Distance cost riskier "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_0025_bins",
                    data_table=table_distance_cost,
                    fields={
                        "value": "Distance cost riskier " + name,
                        "title": "Distance cost riskier " + name,
                    },
                ),
                "TTC cost data "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_005_bins",
                    data_table=table_ttc_cost,
                    fields={
                        "value": "TTC cost data " + name,
                        "title": "TTC cost data " + name,
                    },
                ),
                "TTC cost prediction "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_005_bins",
                    data_table=table_ttc_cost,
                    fields={
                        "value": "TTC cost prediction " + name,
                        "title": "TTC cost prediction " + name,
                    },
                ),
                "TTC cost riskier "
                + name: wandb.plot_table(
                    vega_spec_name="jmercat/histogram_005_bins",
                    data_table=table_ttc_cost,
                    fields={
                        "value": "TTC cost riskier " + name,
                        "title": "TTC cost riskier " + name,
                    },
                ),
            }
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """After a validation at the end of every histogram_interval_epoch,
        log the histograms for two scenes: the safer fast scene and the safer slow scene.
        """
        if (
            trainer.current_epoch % self.histogram_interval_epoch
            == self.histogram_interval_epoch - 1
        ):
            self._log_scene(pl_module, self.scene_safe_fast, name="Safer fast")
            self._log_scene(pl_module, self.scene_safe_slow, name="Safer slow")


class PlotTrajCallback(pl.Callback):
    """Plot trajectory samples for two scenes:
        One that is safer for the slow pedestrians
        One that is safer for the fast pedestrians
    Samples of ground truth, prediction, and biased predictions are superposed.
    Last positions are marked to visualize the clusters.

    Args:
        params: dataclass containing the necessary parameters for a
        n_samples: number of sample trajectories to draw
    """

    def __init__(
        self,
        params: DrawCallbackParams,
        n_samples: int = 1,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.num_steps = params.num_steps
        self.dt = params.scene_params.dt
        self.scene_params = params.scene_params
        self.plot_interval_epoch = params.plot_interval_epoch
        self.scene_safe_fast, self.scene_safe_slow = get_fast_slow_scenes(
            params.scene_params, n_samples
        )
        self.ego_traj = self.scene_safe_fast.get_ego_ref_trajectory(
            params.scene_params.sample_times
        )
        self._risk_sampler = get_risk_level_sampler(params.risk_distribution)

    def _log_scene(self, epoch: int, pl_module, scene: RandomScene, name: str) -> None:
        """Add drawing of samples of prediction, biased prediction and ground truth in the scene.

        Args:
            epoch: current epoch calling the log
            pl_module: pytorch lightning module being trained
            scene: scene to draw
            name: name of the scene
        """
        ped_trajs = scene.get_pedestrians_trajectories()
        device = pl_module.device
        n_agents = ped_trajs.shape[1]

        input_traj = ped_trajs[..., : self.num_steps, :]

        normalized_input, offset = SceneDataLoaders.normalize_trajectory(
            torch.from_numpy(input_traj.astype("float32")).contiguous().to(device)
        )
        mask_input = torch.ones_like(normalized_input[..., 0])
        ego_history = (
            torch.from_numpy(self.ego_traj[..., : self.num_steps, :].astype("float32"))
            .expand_as(normalized_input)
            .contiguous()
            .to(device)
        )
        ego_future = (
            torch.from_numpy(self.ego_traj[..., self.num_steps :, :].astype("float32"))
            .expand(normalized_input.shape[0], n_agents, -1, -1)
            .contiguous()
            .to(device)
        )
        map = torch.empty(ego_history.shape[0], 0, 0, 2, device=mask_input.device)
        mask_map = torch.empty(ego_history.shape[0], 0, 0, device=mask_input.device)

        pred_riskier = (
            pl_module.predict_step(
                (
                    normalized_input,
                    mask_input,
                    map,
                    mask_map,
                    offset,
                    ego_history,
                    ego_future,
                ),
                0,
                risk_level=self._risk_sampler.get_highest_risk(
                    batch_size=self.n_samples, device=device
                )
                .unsqueeze(1)
                .repeat(1, n_agents),
            )
            .cpu()
            .detach()
            .numpy()
        )

        pred = (
            pl_module.predict_step(
                (
                    normalized_input,
                    mask_input,
                    map,
                    mask_map,
                    offset,
                    ego_history,
                    ego_future,
                ),
                0,
                risk_level=None,
            )
            .cpu()
            .detach()
            .numpy()
        )

        fig, ax = plt.subplots()
        plotter = ScenePlotter(scene, ax=ax)
        fig.set_size_inches(h=scene.road_width / 3 + 1, w=scene.road_length / 3)

        time = self.dt * self.num_steps
        plotter.draw_scene(0, time=time)
        alpha = 0.5 / np.log(self.n_samples)
        plotter.draw_all_trajectories(
            ped_trajs[..., self.num_steps :, :],
            color="g",
            alpha=alpha,
            label="Future ground truth",
        )
        plotter.draw_all_trajectories(
            input_traj, color="b", alpha=alpha, label="Past input"
        )
        plotter.draw_all_trajectories(
            pred, color="orange", alpha=alpha, label="Prediction"
        )
        plotter.draw_all_trajectories(
            pred_riskier, color="r", alpha=alpha, label="Prediction risk-seeking"
        )
        plotter.draw_legend()
        plt.tight_layout()
        wandb.log({"Road scene " + name: wandb.Image(fig), "epoch": epoch})
        plt.close()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """After a validation at the end of every plot_interval_epoch,
        log the prediction samples for two scenes: the safer fast scene and the safer slow scene.
        """
        if (
            trainer.current_epoch % self.plot_interval_epoch
            == self.plot_interval_epoch - 1
        ):
            self.scene_safe_fast, self.scene_safe_slow = get_fast_slow_scenes(
                self.scene_params, self.n_samples
            )
            self._log_scene(
                trainer.current_epoch, pl_module, self.scene_safe_slow, "Safer slow"
            )
            self._log_scene(
                trainer.current_epoch, pl_module, self.scene_safe_fast, "Safer fast"
            )


# TODO: make the same kind of logs for the Waymo dataset
