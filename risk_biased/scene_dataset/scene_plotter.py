import os
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams


class ScenePlotter:
    """
    This class defines plotting functions that takes in a scene and an optional axes to plot road agents and trajectories.

    Args:
        scene: The scene to use for plotting
        ax: Matplotlib axes in which the drawing is made
    """

    def __init__(self, scene: RandomScene, ax: Optional[Axes] = None) -> None:
        self.scene = scene
        if ax is None:
            self.ax = plt.subplot()
        else:
            self.ax = ax
        self._sidewalks_boxes = PatchCollection(
            [
                Rectangle(
                    xy=[-scene.ego_length, scene.bottom],
                    height=scene.sidewalks_width,
                    width=scene.road_length + scene.ego_length,
                ),
                Rectangle(
                    xy=[-scene.ego_length, 3 * scene.lane_width / 2],
                    height=scene.sidewalks_width,
                    width=scene.road_length + scene.ego_length,
                ),
            ],
            facecolor="gray",
            alpha=0.3,
            edgecolor="black",
        )
        self._center_line = Line2D(
            [-scene.ego_length / 2, scene.road_length],
            [scene.lane_width / 2, scene.lane_width / 2],
            linewidth=4,
            color="black",
            dashes=[10, 5],
        )

        self._set_agent_patches()
        self._set_agent_paths()
        self.ax.set_aspect("equal")

    def _set_current_time(
        self,
        time: float,
        ego_pos_x: Optional[float] = None,
        ego_pos_y: Optional[float] = None,
    ):
        """
        Set the current time to draw the agents at the proper time along the trajectory.

        Args:
            time: the present time in second
            ego_pos_x (optional): x coordinate of ego position. Defaults to None.
            ego_pos_y (optional): y coordinate of ego position. Defaults to None.
        """
        self.scene.set_current_time(time)
        self._set_agent_patches(ego_pos_x, ego_pos_y)

    def _set_agent_paths(self):
        """
        Defines path as lines.
        """
        self._ego_path = Line2D(
            [0, self.scene.ego_ref_speed * self.scene.time_scene],
            [0, 0],
            linewidth=2,
            color="red",
            dashes=[4, 4],
            alpha=0.3,
        )

        self._pedestrian_path = [
            [
                Line2D(
                    [init[agent, 0], final[agent, 0]],
                    [init[agent, 1], final[agent, 1]],
                    linewidth=2,
                    dashes=[4, 4],
                    alpha=0.3,
                )
                for (init, final) in zip(
                    self.scene.pedestrians_positions,
                    self.scene.final_pedestrians_positions,
                )
            ]
            for agent in range(self.scene.pedestrians_positions.shape[1])
        ]

    def _set_agent_patches(
        self, ego_pos_x: Optional[float] = None, ego_pos_y: Optional[float] = None
    ):
        """
        Set the agent patches at their current position in the scene.

        Args:
            ego_pos_x (optional): x coordinate of ego position. Defaults to None.
            ego_pos_y (optional): y coordinate of ego position. Defaults to None.
        """
        current_step = int(round(self.scene.current_time / self.scene.dt))
        if ego_pos_x is None:
            x = (
                -self.scene.ego_length / 2
                + self.scene.ego_ref_speed * self.scene.current_time
            )
        else:
            x = -self.scene.ego_length / 2 + ego_pos_x
        if ego_pos_y is None:
            y = -self.scene.ego_width / 2
        else:
            y = -self.scene.ego_width / 2 + ego_pos_y
        self._ego_box = Rectangle(
            xy=(x, y),
            height=self.scene.ego_width,
            width=self.scene.ego_length,
            fill=True,
            facecolor="red",
            alpha=0.4,
            edgecolor="black",
        )
        self._pedestrian_patches = [
            [
                Ellipse(
                    xy=xy,
                    width=1,
                    height=0.5,
                    angle=angle * 180 / np.pi + 90,
                    facecolor="blue",
                    alpha=0.4,
                    edgecolor="black",
                )
                for xy, angle in zip(
                    self.scene.pedestrians_trajectories[:, agent, current_step],
                    self.scene.pedestrians.angle[:, agent],
                )
            ]
            for agent in range(self.scene.pedestrians_trajectories.shape[1])
        ]

    def plot_road(self) -> None:
        """
        Plot the road as a two lanes, two sidewalks in straight lines with the ego vehicle. Plot is made in given ax.
        """
        self.ax.add_collection(self._sidewalks_boxes)
        self.ax.add_patch(self._ego_box)
        self.ax.add_line(self._center_line)
        self.ax.add_line(self._ego_path)
        self.rescale()

    def draw_scene(
        self,
        index: int,
        time=None,
        prediction=None,
        ego_pos_x: Optional[float] = None,
        ego_pos_y: Optional[float] = None,
    ) -> None:
        """
        Plot the scene of given index (road, ego vehicle with its path, pedestrian with its path)
        Args:
            index: index of the pedestrian in the batch
            time: set current time to this value if not None
            prediction: draw this instead of the actual future if not None
            ego_pos_x (optional): x coordinate of ego position. Defaults to None.
            ego_pos_y (optional): y coordinate of ego position. Defaults to None.
        """
        if time is not None:
            self._set_current_time(time, ego_pos_x, ego_pos_y)
        self.plot_road()
        for agent_patch in self._pedestrian_patches:
            self.ax.add_patch(agent_patch[index])
        for agent_patch in self._pedestrian_path:
            self.ax.add_line(agent_patch[index])
        if prediction is not None:
            self.draw_trajectory(prediction)

    def rescale(self):
        """
        Set the x and y limits to the road shape with a margin.
        """
        self.ax.set_xlim(
            left=-2 * self.scene.ego_length,
            right=self.scene.road_length + self.scene.ego_length,
        )
        self.ax.set_ylim(
            bottom=self.scene.bottom - self.scene.lane_width,
            top=2 * self.scene.lane_width + 2 * self.scene.sidewalks_width,
        )

    def draw_trajectory(self, prediction, color="b") -> None:
        """
        Plot the given prediction in the scene.
        """
        self.ax.scatter(prediction[..., 0], prediction[..., 1], color=color, alpha=0.3)

    def draw_all_trajectories(
        self,
        prediction: np.ndarray,
        color=None,
        color_value: np.ndarray = None,
        alpha: float = 0.05,
        label: str = "trajectory",
        final_index: int = -1,
    ) -> None:
        """
        Plot all the given predictions in the scene
        Args:
            prediction : (batch, n_agents, time, 2) batch of trajectories
            color: regular color name
            color_value : (batch) Optional batch of values for coloring from green to red
        """

        if color_value is not None:
            min = color_value.min()
            max = color_value.max()
            color_value = 0.9 * (color_value - min) / (max - min)
            for agent in range(prediction.shape[1]):
                for traj, val in zip(prediction[:, agent], color_value[:, agent]):
                    color = (val, 1 - val, 0.1)
                    self.ax.plot(
                        traj[:final_index, 0],
                        traj[:final_index, 1],
                        color=color,
                        alpha=alpha,
                        label=label,
                    )
                    self.ax.scatter(
                        traj[final_index, 0],
                        traj[final_index, 1],
                        color=color,
                        alpha=alpha,
                    )
            cmap = matplotlib.colors.ListedColormap(
                np.linspace(
                    [color_value.min(), 1 - color_value.min(), 0.1],
                    [color_value.max(), 1 - color_value.max(), 0.1],
                    128,
                )
            )
            norm = matplotlib.colors.Normalize(vmin=min, vmax=max, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, label="TTC cost")
        else:
            for agent in range(prediction.shape[1]):
                for traj in prediction:
                    self.ax.plot(
                        traj[agent, :final_index, 0],
                        traj[agent, :final_index, 1],
                        color=color,
                        alpha=alpha,
                        label=label,
                    )
                self.ax.scatter(
                    prediction[:, agent, final_index, 0],
                    prediction[:, agent, final_index, 1],
                    color=color,
                    alpha=alpha,
                )

    def draw_legend(self):
        """Draw legend without repeats and without transparency."""

        handles, labels = self.ax.get_legend_handles_labels()
        i = np.arange(len(labels))
        filter = np.array([])
        unique_labels = list(set(labels))
        for ul in unique_labels:
            filter = np.append(filter, [i[np.array(labels) == ul][0]])
        filtered_handles = []
        for f in filter:
            handles[int(f)].set_alpha(1)
            filtered_handles.append(handles[int(f)])
        filtered_labels = [labels[int(f)] for f in filter]
        self.ax.legend(filtered_handles, filtered_labels)


# Draw a random scene
if __name__ == "__main__":
    from risk_biased.utils.config_argparse import config_argparse

    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    config = config_argparse(config_path)
    n_samples = 100

    scene_params = RandomSceneParams.from_config(config)
    scene_params.batch_size = n_samples
    scene = RandomScene(
        scene_params,
        is_torch=False,
    )

    plotter = ScenePlotter(scene)

    plotter.draw_scene(0, time=1)
    plt.tight_layout()
    plt.show()
