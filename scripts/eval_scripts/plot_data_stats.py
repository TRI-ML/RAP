import math
import os

from mmcv import Config
import matplotlib.pyplot as plt
import numpy as np

from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.cost import (
    DistanceCostNumpy,
    DistanceCostParams,
    TTCCostNumpy,
    TTCCostParams,
)
from risk_biased.utils.risk import get_risk_level_sampler

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )

    config = Config.fromfile(config_path)

    risk_sampler = get_risk_level_sampler(config.risk_distribution)

    is_torch = False
    n_samples = 1000
    sample_every = 10

    scene_params = RandomSceneParams.from_config(config)
    scene_params.batch_size = n_samples
    # Get a batch of random pedestrians
    scene = RandomScene(
        scene_params,
        is_torch=is_torch,
    )

    # Define the initial positions of pedestrians
    # Slow dangerous, fast safe settings:
    percent_right = 0.8
    percent_top = 0.6
    angle = 5 * np.pi / 4
    # Slow safe, fast dangerous settings:
    # percent_right = 0.8
    # percent_top = 1.1
    # angle = 5 * np.pi / 4

    positions = np.array([[[percent_right, percent_top]]] * n_samples)
    angles = np.array([[angle]] * n_samples)
    scene.set_pedestrians_states(positions, angles)

    dist_cost_func = DistanceCostNumpy(DistanceCostParams.from_config(config))
    ttc_cost_func = TTCCostNumpy(TTCCostParams.from_config(config))

    len_traj = int(config.time_scene / scene.dt)
    ped_trajs = scene.get_pedestrians_trajectories()

    ego_traj = scene.get_ego_ref_trajectory(config.sample_times)

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

    fig, ax = plt.subplots()
    plotter = ScenePlotter(scene, ax)
    plotter.draw_scene(0, time=config.num_steps * config.dt)
    # plotter.draw_trajectory(ped_trajs[0, config.num_steps :], color="g")
    plotter.draw_all_trajectories(
        ped_trajs[:, :, config.num_steps :], color_value=ttc_cost
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
        ax[1].hist(dist_cost[:], bins=30)
        ax[1].set_ylim([0, 3 * math.sqrt(n_samples)])
        ax[2].hist(ttc_cost[:], bins=30)
        ax[2].set_ylim([0, 3 * math.sqrt(n_samples)])

    agent_selected = 0
    plot_histograms(travel_distances[:, agent_selected], dist_cost, ttc_cost, "Data")

    print(f"Average ttc cost:      {ttc_cost.mean()}")
    print(f"Average distance cost: {dist_cost.mean()}")

    plt.tight_layout()
    plt.show()
