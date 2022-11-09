import os

import matplotlib.pyplot as plt
from mmcv import Config
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.scene_dataset.scene_plotter import ScenePlotter
from risk_biased.utils.cost import (
    DistanceCostNumpy,
    DistanceCostParams,
    TTCCostNumpy,
    TTCCostParams,
)

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(os.path.join(__file__, "..")))
    config_path = os.path.join(
        working_dir, "..", "risk_biased", "config", "learning_config.py"
    )
    config = Config.fromfile(config_path)
    if config.seed is not None:
        seed_everything(config.seed)
    ped_speed = 2
    is_torch = False

    fig, ax = plt.subplots(
        3, 4, sharex=True, sharey=True, tight_layout=True, subplot_kw={"aspect": 1}
    )

    scene_params = RandomSceneParams.from_config(config)
    scene_params.batch_size = 1
    for ii in range(9):
        test_scene = RandomScene(
            scene_params,
            is_torch=is_torch,
        )
        dist_cost = DistanceCostNumpy(DistanceCostParams.from_config(config))
        ttc_cost = TTCCostNumpy(TTCCostParams.from_config(config))

        nx = 1000
        ny = 100
        x, y = np.meshgrid(
            np.linspace(-test_scene.ego_length, test_scene.road_length, nx),
            np.linspace(test_scene.bottom, test_scene.top, ny),
        )

        i = 2 - (int(ii >= 6) + int(ii >= 3))
        j = ii % 3
        vx = float(ii % 3 - 1)
        vy = float((ii >= 6)) - float(ii <= 2)
        print(f"horizontal velocity {vx}")
        print(f"vertical velocity {vy}")
        norm = np.maximum(np.sqrt(vx * vx + vy * vy), 1)
        vx = vx / norm * np.ones([nx * ny, 1])
        vy = vy / norm * np.ones([nx * ny, 1])
        v_ped = ped_speed * np.stack((vx, vy), -1)
        v_ego = np.array([[[test_scene.ego_ref_speed, 0]]])

        p_init = np.stack((x, y), -1).reshape((nx * ny, 2))
        p_final = p_init + v_ped[:, 0, :] * test_scene.time_scene
        len_traj = 30
        ped_trajs = np.linspace(p_init, p_final, len_traj, axis=1)
        ego_traj = np.linspace(
            [[0, 0]],
            [test_scene.ego_ref_speed * test_scene.time_scene, 0],
            len_traj,
            axis=1,
        )

        cost, _ = ttc_cost(ego_traj, ped_trajs, v_ego, v_ped)

        cost = cost.reshape(ny, nx)
        colorbar = ax[i][j].pcolormesh(x, y, cost, cmap="RdBu_r")
        plotter = ScenePlotter(test_scene, ax=ax[i][j])
        plotter.plot_road()

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    fig.colorbar(colorbar, ax=ax.ravel().tolist())
    for a in ax[:, -1]:
        a.remove()
    plt.show()
