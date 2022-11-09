# Lloyd algorithm while estimating average cost?

import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

# from scipy.cluster.vq import kmeans2
# from scipy.spatial import voronoi_plot_2d, Voronoi
import torch
from torch.utils.data import DataLoader

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import RandomSceneParams

# from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.callbacks import DrawCallbackParams
from risk_biased.utils.config_argparse import config_argparse

from risk_biased.utils.load_model import load_from_config


def draw_travel_distance_map(
    model: torch.nn.Module,
    selected_agent: int,
    loader: DataLoader,
    sqrt_n_samples: int,
    params: DrawCallbackParams,
):
    n_samples = sqrt_n_samples**2
    (
        normalized_input,
        mask_input,
        fut,
        mask_fut,
        mask_loss,
        map,
        mask_map,
        offset,
        ego_past,
        ego_fut,
    ) = next(iter(loader))

    ego_traj = torch.cat((ego_past, ego_fut), dim=2)
    n_scenes, n_agents, n_steps, features = normalized_input.shape
    input_traj = SceneDataLoaders.unnormalize_trajectory(normalized_input, offset)

    # prior_samples = torch.rand(ped_trajs.shape[0], n_samples, 2)*6 - 3
    x = np.linspace(-3, 3, sqrt_n_samples)
    y = np.linspace(-3, 3, sqrt_n_samples)
    xx, yy = np.meshgrid(x, y)

    # Warning: if n_agents>1 the combinations of latent samples are not tested, this is not exploring all the possibilities.
    prior_samples = (
        torch.from_numpy(np.stack((xx, yy), -1).astype("float32"))
        .view(1, 1, n_samples, 2)
        .repeat(n_scenes, n_agents, 1, 1)
    )

    mask_z = torch.ones_like(prior_samples[..., 0, 0])
    y = model.decode(
        z_samples=prior_samples,
        mask_z=mask_z,
        x=normalized_input,
        mask_x=mask_input,
        map=map,
        mask_map=mask_map,
        offset=offset,
    )

    generated_trajs = (
        SceneDataLoaders.unnormalize_trajectory(
            y,
            offset,
        )
        .cpu()
        .detach()
        .numpy()
    )

    # fig, ax = plt.subplots()
    # plotter = ScenePlotter(scene, ax=ax)
    # time = params.scene_params.sample_times[params.num_steps - 1]
    # ind = 0
    # plotter.draw_scene(ind, time=time)
    # plotter.draw_trajectory(input_traj[ind])
    # plotter.draw_all_trajectories(generated_trajs, color="r")
    # plt.show()

    input_traj = np.repeat(
        input_traj.reshape((n_scenes, n_agents, 1, params.num_steps, features)),
        n_samples,
        axis=2,
    )

    generated_ped_trajs = np.concatenate((input_traj, generated_trajs), axis=3)

    travel_distances = np.sqrt(
        np.square(
            generated_ped_trajs[:, :, :, -1] - generated_ped_trajs[:, :, :, 0]
        ).sum(-1)
    )

    travel_distances = (
        travel_distances[:, selected_agent]
        .reshape(n_scenes, sqrt_n_samples, sqrt_n_samples)
        .mean(0)
    )
    cmap = plt.get_cmap("RdBu_r")
    vmin = params.scene_params.time_scene * params.scene_params.slow_speed
    vmax = params.scene_params.time_scene * params.scene_params.fast_speed
    plt.contourf(
        xx,
        yy,
        travel_distances,
        50,
        cmap=cmap,
        extent=(-3, 3, -3, 3),
        vmin=vmin,
        vmax=vmax,
    )
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Travel distance")
    plt.axis([-3, 3, -3, 3])
    plt.show()


if __name__ == "__main__":
    # Draws a map, in the latent space, of travel distances averaged on a batch of input trajectories.
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = config_argparse(config_path)

    cfg.batch_size = 128
    model, loaders, cfg = load_from_config(cfg)
    assert (
        cfg.latent_dim == 2
        and "The latent dimension of the model must be exactly 2 to be plotted (no dimensionality reduction capabilities)"
    )
    scene_params = RandomSceneParams.from_config(cfg)
    draw_params = DrawCallbackParams.from_config(cfg)
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    sqrt_n_samples = 20
    draw_travel_distance_map(
        model.model,
        0,
        loaders.val_dataloader(),
        sqrt_n_samples,
        draw_params,
    )
