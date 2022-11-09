import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.utils.data import DataLoader

from risk_biased.scene_dataset.scene import RandomSceneParams
from matplotlib.patches import Ellipse
from risk_biased.utils.callbacks import DrawCallbackParams
from risk_biased.utils.config_argparse import config_argparse

from risk_biased.utils.load_model import load_from_config

from scripts.scripts_utils.sample_batch_utils import repeat_and_reshape_all


def draw_latent_biased(
    model: torch.nn.Module, device, loader: DataLoader, params: DrawCallbackParams
):
    n_samples = 9
    # ped_trajs = scene.get_pedestrians_trajectories()
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
    ) = repeat_and_reshape_all(next(iter(loader)), n_samples)
    n_scenes_samples, n_agents, n_steps, features = normalized_input.shape
    n_scenes = n_scenes_samples // n_samples

    risk_level = torch.ones(n_scenes, n_samples, n_agents) * torch.linspace(
        0, 1, n_samples
    ).unsqueeze(0).unsqueeze(-1)

    y_sample, biased_mu, biased_log_std = model(
        normalized_input,
        mask_input,
        map,
        mask_map,
        offset=offset,
        x_ego=ego_past,
        y_ego=ego_fut,
        risk_level=risk_level.view(-1, n_agents),
    )

    biased_mu = biased_mu.view(n_scenes, n_agents, n_samples, 2).cpu().detach().numpy()
    biased_std = (
        biased_log_std.view(n_scenes, n_agents, n_samples, 2)
        .exp()
        .cpu()
        .detach()
        .numpy()
    )
    risk_level = risk_level.permute(0, 2, 1).cpu().detach().numpy()

    fig, ax = plt.subplots(3, 3)
    cmap = plt.get_cmap("RdBu_r")
    for s in range(n_samples):
        for i in range(n_scenes):
            for a in range(n_agents):
                ii = s // 3
                jj = s % 3
                ellipse = Ellipse(
                    (biased_mu[i, a, s, 0], biased_mu[i, a, s, 1]),
                    width=biased_std[i, a, s, 0] * 2,
                    height=biased_std[i, a, s, 1] * 2,
                    facecolor="none",
                    edgecolor=(*cmap(risk_level[i, a, s])[:-1], 0.05),
                )
                ax[ii][jj].add_patch(ellipse)
                ax[ii][jj].axis([-3, 3, -3, 3])

    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax.ravel().tolist(), label="Desired risk levels")
    plt.show()


if __name__ == "__main__":
    # Draws 9 plots in the latent space. In each plot a constant risk level is used (0 for plot 0 and 1 for plot 8).
    # Each plot superposes ellipses representing the encoded distributions for a batch of x input.
    # Each plot represents the latent distribution for the same batch of input but at different risk levels.
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    cfg = config_argparse(config_path)

    cfg.batch_size = cfg.datasets_sizes["val"]
    model, loaders, cfg = load_from_config(cfg)
    assert (
        cfg.latent_dim == 2
        and "The latent dimension of the model must be exactly 2 to be plotted (no dimensionality reduction capabilities)"
    )
    scene_params = RandomSceneParams.from_config(cfg)
    scene = loaders.val_dataloader()
    draw_params = DrawCallbackParams.from_config(cfg)
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    draw_latent_biased(model.model, model.device, scene, draw_params)
