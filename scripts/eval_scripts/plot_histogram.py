import math
import os

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import torch

from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.utils.cost import (
    DistanceCostNumpy,
    DistanceCostParams,
    TTCCostNumpy,
    TTCCostParams,
)
from risk_biased.utils.load_model import load_from_config
from risk_biased.utils.risk import get_risk_level_sampler
from risk_biased.utils.config_argparse import config_argparse

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    config = config_argparse(config_path)
    model, loaders, config = load_from_config(config)
    if config.seed is not None:
        seed_everything(config.seed)

    risk_sampler = get_risk_level_sampler(config.risk_distribution)

    is_torch = False
    n_scenes = 1000
    sample_every = 10

    # Get a batch of random pedestrians
    scene_params = RandomSceneParams.from_config(config)
    scene_params.batch_size = n_scenes
    scene = RandomScene(
        scene_params,
        is_torch=is_torch,
    )

    dist_cost_func = DistanceCostNumpy(DistanceCostParams.from_config(config))
    ttc_cost_func = TTCCostNumpy(TTCCostParams.from_config(config))

    len_traj = int(config.time_scene / scene.dt)
    ped_trajs = scene.get_pedestrians_trajectories()
    ped_trajs_past = ped_trajs[:, :, : config.num_steps]

    batch_size = ped_trajs.shape[0]
    ego_traj = scene.get_ego_ref_trajectory(config.sample_times).repeat(
        batch_size, axis=0
    )

    normalized_trajs, offset = loaders.normalize_trajectory(
        torch.from_numpy(ped_trajs.astype("float32")).contiguous()
    )
    x = normalized_trajs[:, :, : config.num_steps]
    ego_history = (
        torch.from_numpy(ego_traj[:, :, : config.num_steps].astype("float32"))
        .expand_as(x)
        .contiguous()
    )
    ego_future = (
        torch.from_numpy(ego_traj[:, :, -config.num_steps_future :].astype("float32"))
        .expand(x.shape[0], x.shape[1], -1, -1)
        .contiguous()
    )
    mask_x = torch.ones_like(x[..., 0])
    map = torch.empty(ego_history.shape[0], 0, 0, 2, device=mask_x.device)
    mask_map = torch.empty(ego_history.shape[0], 0, 0, device=mask_x.device)

    pred_riskier = (
        model.predict_step(
            (x, mask_x, map, mask_map, offset, ego_history, ego_future),
            0,
            risk_level=risk_sampler.get_highest_risk(
                batch_size=n_scenes, device="cpu"
            ).unsqueeze(1),
        )
        .cpu()
        .detach()
        .numpy()
    )

    pred = (
        model.predict_step(
            (x, mask_x, map, mask_map, offset, ego_history, ego_future),
            0,
            risk_level=None,
        )
        .cpu()
        .detach()
        .numpy()
    )

    ped_trajs_pred = np.concatenate((ped_trajs_past, pred), axis=-2)
    ped_trajs_pred_riskier = np.concatenate((ped_trajs_past, pred_riskier), axis=-2)

    travel_distances = np.sqrt(
        np.square(ped_trajs[..., -1, :] - ped_trajs[..., 0, :]).sum(-1)
    )

    dist_cost, dist = dist_cost_func(
        ego_traj[:, :, config.num_steps :], ped_trajs[:, :, config.num_steps :]
    )

    ttc_cost, (ttc, dist) = ttc_cost_func(
        ego_traj[:, :, config.num_steps :],
        ped_trajs[:, :, config.num_steps :],
        scene.get_ego_ref_velocity(),
        scene.get_pedestrians_velocities(),
    )

    travel_distances_pred = np.sqrt(
        np.square(ped_trajs_pred[..., -1, :] - ped_trajs_pred[..., 0, :]).sum(-1)
    )
    dist_cost_pred, dist_pred = dist_cost_func(
        ego_traj[:, :, config.num_steps :], ped_trajs_pred[:, :, config.num_steps :]
    )
    sample_times = np.array(config.sample_times)
    ped_velocities_pred = (ped_trajs_pred[:, :, 1:] - ped_trajs_pred[:, :, :-1]) / (
        (sample_times[1:] - sample_times[:-1])[None, None, :, None]
    )
    ped_velocities_pred = np.concatenate(
        (ped_velocities_pred[:, :, 0:1], ped_velocities_pred), -2
    )
    ttc_cost_pred, (ttc_pred, dist_pred) = ttc_cost_func(
        ego_traj[:, :, config.num_steps :],
        ped_trajs_pred[:, :, config.num_steps :],
        scene.get_ego_ref_velocity(),
        ped_velocities_pred[:, :, config.num_steps :],
    )

    travel_distances_pred_riskier = np.sqrt(
        np.square(
            ped_trajs_pred_riskier[..., -1, :] - ped_trajs_pred_riskier[..., 0, :]
        ).sum(-1)
    )

    dist_cost_pred_riskier, dist_pred_riskier = dist_cost_func(
        ego_traj[:, :, config.num_steps :],
        ped_trajs_pred_riskier[:, :, config.num_steps :],
    )
    sample_times = np.array(config.sample_times)
    ped_velocities_pred_riskier = (
        ped_trajs_pred_riskier[:, :, 1:] - ped_trajs_pred_riskier[:, :, :-1]
    ) / ((sample_times[1:] - sample_times[:-1])[None, None, :, None])
    ped_velocities_pred_riskier = np.concatenate(
        (ped_velocities_pred_riskier[:, :, 0:1], ped_velocities_pred_riskier), 2
    )
    ttc_cost_pred_riskier, (ttc_pred, dist_pred_riskier) = ttc_cost_func(
        ego_traj[:, :, config.num_steps :],
        ped_trajs_pred_riskier[:, :, config.num_steps :],
        scene.get_ego_ref_velocity(),
        ped_velocities_pred_riskier[:, :, config.num_steps :],
    )

    def plot_histograms(travel_distances, dist_cost, ttc_cost, label=""):
        # Open the plots for the sampled future times
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(label)

        # Plot histograms of traveled distances, depending on the parameters.
        # It should be multi-modal. There is a minimum distance and a maximum distance and travel distance variations within these bounds.
        ax[0].set_title("Travel distance")
        ax[1].set_title("Distance cost")
        ax[2].set_title("TTC cost")

        ax[0].hist(travel_distances, bins=30)
        ax[1].hist(dist_cost.flatten(), bins=30)
        ax[1].set_ylim([0, 3 * math.sqrt(n_scenes)])
        ax[2].hist(ttc_cost.flatten(), bins=30)
        ax[2].set_ylim([0, 3 * math.sqrt(n_scenes)])

    agent_selected = 0

    plot_histograms(
        travel_distances[:, agent_selected],
        dist_cost[:, agent_selected],
        ttc_cost[:, agent_selected],
        "Data",
    )
    plot_histograms(
        travel_distances_pred[:, agent_selected],
        dist_cost_pred[:, agent_selected],
        ttc_cost_pred[:, agent_selected],
        "Prediction normal risk",
    )
    plot_histograms(
        travel_distances_pred_riskier[:, agent_selected],
        dist_cost_pred_riskier[:, agent_selected],
        ttc_cost_pred_riskier[:, agent_selected],
        "Prediction high risk",
    )

    print(f"Average ttc risk")
    print(
        f"Ground truth: {ttc_cost.mean()}, Prediction: {ttc_cost_pred.mean()}, Biased prediction: {ttc_cost_pred_riskier.mean()}"
    )

    plt.show()
