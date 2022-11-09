# Lloyd algorithm while estimating average cost?

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from pytest import param
from pytorch_lightning.utilities.seed import seed_everything
import torch

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from matplotlib.patches import Ellipse
from risk_biased.utils.callbacks import get_fast_slow_scenes, DrawCallbackParams
from risk_biased.utils.config_argparse import config_argparse

from risk_biased.utils.load_model import load_from_config


def draw_latent_biased(
    model: torch.nn.Module,
    device,
    scene: RandomScene,
    n_samples: int,
    params: DrawCallbackParams,
):
    ped_trajs = scene.get_pedestrians_trajectories()
    n_scenes, n_agents, n_steps, features = ped_trajs.shape
    ego_traj = scene.get_ego_ref_trajectory([t * params.dt for t in range(n_steps)])
    ego_past, ego_future = torch.split(
        torch.from_numpy(ego_traj.astype("float32")),
        [params.num_steps, params.num_steps_future],
        dim=2,
    )
    ego_past = ego_past.repeat(n_scenes, n_samples, 1, 1).view(
        -1, 1, params.num_steps, features
    )
    ego_future = ego_future.repeat(n_scenes, n_samples, 1, 1).view(
        -1, 1, params.num_steps_future, features
    )
    input_traj = ped_trajs[:, :, : params.num_steps]
    normalized_input, offset = SceneDataLoaders.normalize_trajectory(
        torch.from_numpy(input_traj.astype("float32")).contiguous().to(device)
    )
    normalized_input = (
        normalized_input.view(n_scenes, 1, n_agents, params.num_steps, features)
        .repeat(1, n_samples, 1, 1, 1)
        .view(-1, n_agents, params.num_steps, features)
    )
    mask_input = torch.ones_like(normalized_input[..., 0])
    map = torch.empty(n_scenes, 0, 0, features, device=device)
    mask_map = torch.empty(n_scenes, 0, 0)
    offset = (
        offset.view(n_scenes, 1, n_agents, features)
        .repeat(1, n_samples, 1, 1)
        .view(-1, n_agents, features)
    )
    n_scenes = ped_trajs.shape[0]

    risk_level = (
        torch.linspace(0, 1, n_samples)
        .view(1, n_samples, 1)
        .repeat(n_scenes, 1, n_agents)
    )

    y_samples, biased_mu, biased_log_std = model(
        normalized_input,
        mask_input,
        map,
        mask_map,
        offset=offset,
        x_ego=ego_past,
        y_ego=ego_future,
        risk_level=risk_level.view(-1, n_agents),
    )

    biased_mu = (
        biased_mu.reshape(n_scenes, n_samples, n_agents, 2)
        .permute(0, 2, 1, 3)
        .mean(0)
        .cpu()
        .detach()
        .numpy()
    )
    biased_std = (
        (
            2
            * biased_log_std.reshape(n_scenes, n_samples, n_agents, 2).permute(
                0, 2, 1, 3
            )
        )
        .exp()
        .mean(0)
        .sqrt()
        .cpu()
        .detach()
        .numpy()
    )
    risk_level = risk_level.permute(0, 2, 1).mean(0).cpu().detach().numpy()

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("RdBu_r")
    for a in range(n_agents):
        for j in range(n_samples):
            ellipse = Ellipse(
                (biased_mu[a, j, 0], biased_mu[a, j, 1]),
                width=biased_std[a, j, 0] * 2,
                height=biased_std[a, j, 1] * 2,
                facecolor=cmap(risk_level[a, j]),
                alpha=0.1,
            )
            ax.add_patch(ellipse)
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label="Risk level")
    plt.axis([-3, 3, -3, 3])
    plt.show()


if __name__ == "__main__":
    # Draws the biased distributions in latent space for different risk levels in two scenarios: safer_fast and safer_slow.
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
    safer_fast_scene, safer_slow_scene = get_fast_slow_scenes(scene_params, 128)
    draw_params = DrawCallbackParams.from_config(cfg)
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    n_samples = 64
    draw_latent_biased(
        model.model, model.device, safer_fast_scene, n_samples, draw_params
    )
    draw_latent_biased(
        model.model, model.device, safer_slow_scene, n_samples, draw_params
    )
