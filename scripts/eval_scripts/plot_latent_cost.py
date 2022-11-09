# Lloyd algorithm while estimating average cost?

import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

# from scipy.cluster.vq import kmeans2
# from scipy.spatial import voronoi_plot_2d, Voronoi
import torch
import torch.nn as nn

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams

# from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.callbacks import get_fast_slow_scenes, DrawCallbackParams
from risk_biased.utils.config_argparse import config_argparse
from risk_biased.utils.cost import TTCCostNumpy

from risk_biased.utils.load_model import load_from_config


def draw_cost_map(
    model: nn.Module,
    selected_agent: int,
    device,
    scene: RandomScene,
    sqrt_n_samples: int,
    params: DrawCallbackParams,
):
    n_samples = sqrt_n_samples**2
    ped_trajs = scene.get_pedestrians_trajectories()
    n_scenes, n_agents, n_steps, features = ped_trajs.shape
    input_traj = ped_trajs[:, :, : params.num_steps]
    normalized_input, offset = SceneDataLoaders.normalize_trajectory(
        torch.from_numpy(input_traj.astype("float32")).contiguous().to(device)
    )

    n_scenes = ped_trajs.shape[0]

    x = np.linspace(-3, 3, sqrt_n_samples)
    y = np.linspace(-3, 3, sqrt_n_samples)
    xx, yy = np.meshgrid(x, y)
    prior_samples = (
        torch.from_numpy(np.stack((xx, yy), -1).astype("float32"))
        .view(1, 1, n_samples, 2)
        .repeat(n_scenes, n_agents, 1, 1)
    )

    mask_z = torch.ones_like(prior_samples[..., 0, 0])
    mask_input = torch.ones_like(normalized_input[..., 0])
    map = torch.empty(n_scenes, 0, 0, features, device=device)
    mask_map = torch.empty(n_scenes, 0, 0)
    generated_trajs = (
        SceneDataLoaders.unnormalize_trajectory(
            model.decode(
                z_samples=prior_samples,
                mask_z=mask_z,
                x=normalized_input,
                mask_x=mask_input,
                map=map,
                mask_map=mask_map,
                offset=offset,
            ),
            offset,
        )
        .cpu()
        .detach()
        .numpy()
    )

    input_traj = np.repeat(
        input_traj.reshape((n_scenes, n_agents, 1, params.num_steps, features)),
        n_samples,
        axis=2,
    )

    generated_ped_trajs = np.concatenate((input_traj, generated_trajs), axis=3)
    ego_traj = scene.get_ego_ref_trajectory(params.scene_params.sample_times)[
        None, :, :
    ]
    ttc_cost_func = TTCCostNumpy(params.ttc_cost_params)

    sample_times = np.array(params.scene_params.sample_times)
    ped_velocities = (
        generated_ped_trajs[:, :, :, 1:] - generated_ped_trajs[:, :, :, :-1]
    ) / ((sample_times[1:] - sample_times[:-1])[None, None, None, :, None])
    ped_velocities = np.concatenate((ped_velocities[:, :, :, 0:1], ped_velocities), 3)
    ttc_cost_pred, (ttc_pred, dist_pred) = ttc_cost_func(
        ego_traj[:, :, :, params.num_steps :],
        generated_ped_trajs[:, :, :, params.num_steps :],
        scene.get_ego_ref_velocity()[:, :, None],
        ped_velocities[:, :, :, params.num_steps :],
    )

    ttc_cost_pred = (
        ttc_cost_pred[:, selected_agent]
        .reshape(n_scenes, sqrt_n_samples, sqrt_n_samples)
        .mean(0)
    )
    cmap = plt.get_cmap("RdBu_r")
    plt.contourf(
        xx,
        yy,
        ttc_cost_pred.reshape((sqrt_n_samples, sqrt_n_samples)),
        50,
        cmap=cmap,
        extent=(-3, 3, -3, 3),
        vmin=0,
        vmax=2,
    )
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="TTC cost")
    plt.axis([-3, 3, -3, 3])
    plt.show()


if __name__ == "__main__":
    # Draws a contour plot of the cost associated with the latent samples in two scenarios: safer_fast and safer_slow
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = config_argparse(config_path)

    model, loaders, cfg = load_from_config(cfg)
    assert (
        cfg.latent_dim == 2
        and "The latent dimension of the model must be exactly 2 to be plotted (no dimensionality reduction capabilities)"
    )
    scene_params = RandomSceneParams.from_config(cfg)
    safer_fast_scene, safer_slow_scene = get_fast_slow_scenes(scene_params, 100)
    draw_params = DrawCallbackParams.from_config(cfg)
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    sqrt_n_samples = 100
    n_quantize = 100
    draw_cost_map(
        model.model,
        0,
        model.device,
        safer_fast_scene,
        sqrt_n_samples,
        draw_params,
    )
    draw_cost_map(
        model.model,
        0,
        model.device,
        safer_slow_scene,
        sqrt_n_samples,
        draw_params,
    )
