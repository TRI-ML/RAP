import os

from mmcv import Config
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import torch

from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.cost import (
    DistanceCostTorch,
    DistanceCostParams,
    TTCCostTorch,
    TTCCostParams,
)
from risk_biased.utils.load_model import load_from_config
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

    is_torch = True
    n_scenes = 100
    risk_level = 0
    fig, ax = plt.subplots()

    scene_params = RandomSceneParams.from_config(config)
    scene_params.batch_size = n_scenes
    test_scene = RandomScene(
        scene_params,
        is_torch=is_torch,
    )
    plotter = ScenePlotter(test_scene, ax=ax)
    num_steps = config.num_steps
    time = config.sample_times[config.num_steps - 1]

    dist_cost = DistanceCostTorch(DistanceCostParams.from_config(config))
    ttc_cost = TTCCostTorch(TTCCostParams.from_config(config))

    len_traj = int(config.time_scene / test_scene.dt)
    ped_trajs = test_scene.get_pedestrians_trajectories()[
        :, :, [int(round(t / config.dt)) for t in config.sample_times]
    ]

    ego_traj = test_scene.get_ego_ref_trajectory(config.sample_times)

    batch_size = ped_trajs.shape[0]
    normalized_trajs, offset = loaders.normalize_trajectory(ped_trajs)
    x = normalized_trajs[:, :, : config.num_steps]
    ego_history = ego_traj[:, :, : config.num_steps].repeat(batch_size, 1, 1, 1)
    ego_future = ego_traj[:, :, config.num_steps :].repeat(batch_size, 1, 1, 1)
    mask_x = torch.ones_like(x[..., 0])
    map = torch.empty(ego_history.shape[0], 0, 0, 2, device=mask_x.device)
    mask_map = torch.empty(ego_history.shape[0], 0, 0, device=mask_x.device)
    # ego_conditioning = model.get_ego_conditioning(ego_history, ego_future)
    pred = (
        model.predict_step(
            (x, mask_x, map, mask_map, offset, ego_history, ego_future),
            0,
            risk_level=torch.ones(n_scenes, 1, device=x.device) * risk_level,
        )
        .cpu()
        .detach()
        .numpy()
    )

    text_length = 10
    text_height = 1

    ind = int(np.random.rand() * n_scenes)
    agent_selected = 0

    plotter.draw_scene(ind, time=time)
    plotter.draw_trajectory(
        ped_trajs[ind, agent_selected, config.num_steps :], color="g"
    )
    plotter.draw_trajectory(ped_trajs[ind, agent_selected, : config.num_steps])
    plotter.draw_trajectory(pred[ind, agent_selected], color="r")

    ped_velocities = test_scene.get_pedestrians_velocities().repeat(
        (1, 1, ped_trajs.shape[2], 1)
    )
    cost, (ttc, dist) = ttc_cost(
        ego_traj[:, :, config.num_steps :],
        ped_trajs[:, :, config.num_steps :],
        test_scene.get_ego_ref_velocity(),
        ped_velocities[:, :, config.num_steps :],
    )
    print(f"Equation TTC: {ttc[ind, agent_selected, num_steps]:.2f}")
    print(f"Distance at TTC {dist[ind, agent_selected, num_steps]:.2f}")
    plt.text(
        test_scene.road_length - text_length,
        test_scene.road_width - 2 * text_height,
        f"TTC cost: {cost[ind, agent_selected]:.2f}",
    )
    cost, dist = dist_cost(
        ego_traj[:, :, config.num_steps :], ped_trajs[:, :, config.num_steps :]
    )
    cost = cost[ind, agent_selected]
    if is_torch:
        print(
            f"Min distance {torch.sqrt(torch.min(dist, 2)[0][ind, agent_selected]):.2f}"
        )
    else:
        print(f"Min distance {np.sqrt(np.min(dist, 2)[ind, agent_selected]):.2f}")
    ax.text(
        test_scene.road_length - text_length,
        test_scene.road_width - 3 * text_height,
        f"Distance cost: {cost:.2f}",
    )
    plt.tight_layout()
    plt.show()
