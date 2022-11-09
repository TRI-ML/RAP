"""evaluate_prediction_planning_stack.py --load_from <wandb ID> --seed <seed>
--num_episodes <num_episodes> --risk_level <a list of risk-levels>
--num_samples <a list of numbers of prediction samples> --load_last --force_config

This script loads a trained predictor from <wand ID>, runs a batch of open-loop MPC evaluations
(i.e., without replanning) using <num_episodes> episodes while varying risk-levels and numbers of
prediction samples. Results are stored in scripts/logs/planner_eval/run-<wandb ID>_<seed> as a
collection of pickle files.
"""


import argparse
import os
import pickle
from time import time_ns
from typing import List, Tuple
import sys

import torch
from mmcv import Config
from pytorch_lightning import seed_everything
from tqdm import trange


from risk_biased.mpc_planner.planner import MPCPlanner, MPCPlannerParams
from risk_biased.predictors.biased_predictor import LitTrajectoryPredictor
from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import RandomScene, RandomSceneParams
from risk_biased.utils.callbacks import get_fast_slow_scenes
from risk_biased.utils.load_model import load_from_config, config_argparse
from risk_biased.utils.planner_utils import (
    evaluate_control_sequence,
    get_interaction_cost,
    AbstractState,
    to_state,
)


def evaluate_main(
    load_from: str,
    seed: int,
    num_episodes: int,
    risk_level_list: List[float],
    num_prediction_samples_list: List[int],
):
    print(f"Risk-sensitivity level(s) to test: {risk_level_list}")
    print(f"Number(s) of prediction samples to test: {num_prediction_samples_list} ")
    save_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "logs",
        "planner_eval",
        f"run-{load_from}_{seed}",
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg, planner = get_cfg_and_planner(load_from=load_from)
    if not planner.solver.params.mean_warm_start == False:
        print(
            "switching to mean_warm_start = False for open-loop evaluation (i.e. without re-planning)"
        )
        planner.solver.params.mean_warm_start = False

    for scene_type in [
        "safer_slow",
        "safer_fast",
    ]:
        print(f"\nTesting {scene_type} scenes")
        seed_everything(seed)
        (
            scene,
            ado_state_history_batch,
            ado_state_future_batch,
        ) = get_scene_and_ado_trajectory(
            cfg, scene_type=scene_type, num_episodes=num_episodes
        )

        (
            ego_state_history,
            ego_state_target_trajectory,
        ) = get_ego_state_history_and_target_trajectory(cfg, scene)

        for stack_risk_level in risk_level_list:
            print(f"  Risk_level: {stack_risk_level}")
            file_name = f"{scene_type}_no_policy_opt_risk_level_{stack_risk_level}"
            print(f"    {file_name}")
            stats_dict_no_policy_opt = evaluate_prediction_planning_stack(
                planner,
                ado_state_history_batch,
                ado_state_future_batch,
                ego_state_history,
                ego_state_target_trajectory,
                optimize_policy=False,
                stack_risk_level=stack_risk_level,
                risk_in_predictor=False,
            )
            with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
                pickle.dump(stats_dict_no_policy_opt, f)

            for num_prediction_samples in num_prediction_samples_list:
                file_name = f"{scene_type}_{num_prediction_samples}_samples_risk_level_{stack_risk_level}"
                if stack_risk_level == 0.0:
                    print(f"    {file_name}")
                    stats_dict_risk_neutral = evaluate_prediction_planning_stack(
                        planner,
                        ado_state_history_batch,
                        ado_state_future_batch,
                        ego_state_history,
                        ego_state_target_trajectory,
                        optimize_policy=True,
                        stack_risk_level=stack_risk_level,
                        risk_in_predictor=False,
                        num_prediction_samples=num_prediction_samples,
                    )
                    with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
                        pickle.dump(stats_dict_risk_neutral, f)
                else:
                    file_name_in_predictor = file_name + "_in_predictor"
                    print(f"    {file_name_in_predictor}")
                    stats_dict_risk_in_predictor = evaluate_prediction_planning_stack(
                        planner,
                        ado_state_history_batch,
                        ado_state_future_batch,
                        ego_state_history,
                        ego_state_target_trajectory,
                        optimize_policy=True,
                        stack_risk_level=stack_risk_level,
                        risk_in_predictor=True,
                        num_prediction_samples=num_prediction_samples,
                    )
                    with open(
                        os.path.join(save_dir, file_name_in_predictor + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(stats_dict_risk_in_predictor, f)
                    file_name_in_planner = file_name + "_in_planner"
                    print(f"    {file_name_in_planner}")
                    stats_dict_risk_in_planner = evaluate_prediction_planning_stack(
                        planner,
                        ado_state_history_batch,
                        ado_state_future_batch,
                        ego_state_history,
                        ego_state_target_trajectory,
                        optimize_policy=True,
                        stack_risk_level=stack_risk_level,
                        risk_in_predictor=False,
                        num_prediction_samples=num_prediction_samples,
                    )
                    with open(
                        os.path.join(save_dir, file_name_in_planner + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(stats_dict_risk_in_planner, f)


def evaluate_prediction_planning_stack(
    planner: MPCPlanner,
    ado_state_history_batch: AbstractState,
    ado_state_future_batch: AbstractState,
    ego_state_history: AbstractState,
    ego_state_target_trajectory: AbstractState,
    optimize_policy: bool = True,
    stack_risk_level: float = 0.0,
    risk_in_predictor: bool = False,
    num_prediction_samples: int = 128,
    num_prediction_samples_for_policy_eval: int = 4096,
) -> dict:
    assert planner.solver.params.mean_warm_start == False
    if risk_in_predictor:
        predictor_risk_level, planner_risk_level = stack_risk_level, 0.0
    else:
        predictor_risk_level, planner_risk_level = 0.0, stack_risk_level

    stats_dict = {
        "stack_risk_level": stack_risk_level,
        "predictor_risk_level": predictor_risk_level,
        "planner_risk_level": planner_risk_level,
    }

    num_episodes = ado_state_history_batch.shape[0]
    assert num_episodes == ado_state_future_batch.shape[0]

    for episode_id in trange(num_episodes, desc="episodes", leave=False):
        ado_state_history = ado_state_history_batch[episode_id]
        ado_state_future = ado_state_future_batch[episode_id]

        (
            ado_state_future_samples_for_policy_eval,
            sample_weights,
        ) = planner.solver.sample_prediction(
            planner.predictor,
            ado_state_history,
            planner.normalizer,
            ego_state_history=ego_state_history,
            ego_state_future=ego_state_target_trajectory,
            num_prediction_samples=num_prediction_samples_for_policy_eval,
            risk_level=0.0,
        )

        if optimize_policy:
            start = time_ns()
            solver_info = planner.solver.solve(
                planner.predictor,
                ego_state_history,
                ego_state_target_trajectory,
                ado_state_history,
                planner.normalizer,
                num_prediction_samples=num_prediction_samples,
                verbose=False,
                risk_level=stack_risk_level,
                resample_prediction=False,
                risk_in_predictor=risk_in_predictor,
            )
            end = time_ns()
            computation_time_ms = (end - start) * 1e-6
        else:
            planner.solver.reset()
            computation_time_ms = 0.0
            solver_info = None

        interaction_cost_gt = get_ground_truth_interaction_cost(
            planner, ado_state_future, ego_state_history
        )

        interaction_risk, tracking_cost = evaluate_control_sequence(
            planner.solver.control_sequence,
            planner.solver.dynamics_model,
            ego_state_history,
            ego_state_target_trajectory,
            ado_state_future_samples_for_policy_eval,
            sample_weights,
            planner.solver.interaction_cost,
            planner.solver.tracking_cost,
            risk_level=stack_risk_level,
            risk_estimator=planner.solver.risk_estimator,
        )

        stats_dict_this_run = {
            "computation_time_ms": computation_time_ms,
            "interaction_cost_ground_truth": interaction_cost_gt,
            "interaction_risk": interaction_risk,
            "tracking_cost": tracking_cost,
            "control_sequence": planner.solver.control_sequence,
            # "solver_info": solver_info,
            # "ado_unbiased_predictions": ado_state_future_samples_for_policy_eval.position.detach()
            # .cpu()
            # .numpy(),
            "sample_weights": sample_weights.detach().cpu().numpy(),
            "ado_position_future": ado_state_future.position.detach().cpu().numpy(),
            "ado_position_history": ado_state_history.position.detach().cpu().numpy(),
        }
        stats_dict[episode_id] = stats_dict_this_run

    return stats_dict


def get_cfg_and_predictor() -> Tuple[Config, LitTrajectoryPredictor]:
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "risk_biased",
        "config",
        "learning_config.py",
    )
    cfg = config_argparse(config_path)
    predictor, _, cfg = load_from_config(cfg)
    return cfg, predictor


def get_cfg_and_planner(load_from: str) -> Tuple[Config, MPCPlanner]:
    planner_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "risk_biased",
        "config",
        "planning_config.py",
    )
    planner_cfg = Config.fromfile(planner_config_path)
    cfg, predictor = get_cfg_and_predictor()

    # joint_dict = {**dict(cfg), **dict(planner_cfg)}
    # assert joint_dict == {
    #     **dict(planner_cfg),
    #     **dict(cfg),
    # }, f"some of the entries conflict between {cfg.filename} and {planner_cfg.filename}"
    # joint_cfg = Config(joint_dict)
    cfg.update(planner_cfg)

    planner_params = MPCPlannerParams.from_config(cfg)
    normalizer = SceneDataLoaders.normalize_trajectory

    planner = MPCPlanner(planner_params, predictor, normalizer)
    return cfg, planner


def get_scene_and_ado_trajectory(
    cfg: Config, scene_type: str, num_episodes: int
) -> Tuple[RandomScene, torch.Tensor, torch.Tensor]:
    scene_params = RandomSceneParams.from_config(cfg)
    safer_fast_scene, safer_slow_scene = get_fast_slow_scenes(
        scene_params, num_episodes
    )

    assert scene_type in ["safer_fast", "safer_slow"]
    if scene_type == "safer_fast":
        scene = safer_fast_scene
    elif scene_type == "safer_slow":
        scene = safer_slow_scene
    else:
        raise ValueError(f"unknown scene type {scene_type}")

    ado_trajectory = torch.from_numpy(
        scene.get_pedestrians_trajectories().astype("float32")
    )
    ado_position_history = to_state(ado_trajectory[..., : cfg.num_steps, :], cfg.dt)
    ado_position_future = to_state(ado_trajectory[..., cfg.num_steps :, :], cfg.dt)
    return scene, ado_position_history, ado_position_future


def get_ego_state_history_and_target_trajectory(
    cfg: Config, scene: RandomScene
) -> Tuple[torch.Tensor, torch.Tensor]:
    ego_state_traj = to_state(
        torch.from_numpy(
            scene.get_ego_ref_trajectory(cfg.sample_times).astype("float32")
        ),
        cfg.dt,
    )
    ego_state_history = ego_state_traj[0, :, : cfg.num_steps]
    ego_state_target_trajectory = ego_state_traj[0, :, cfg.num_steps :]
    return ego_state_history, ego_state_target_trajectory


def get_ground_truth_interaction_cost(
    planner: MPCPlanner,
    ado_state_future: AbstractState,  # (num_agents, num_steps_future)
    ego_state_history: AbstractState,  # (1, 1, num_steps)
) -> float:
    ego_state_future = planner.solver.dynamics_model.simulate(
        ego_state_history[..., -1], planner.solver.control_sequence.unsqueeze(0)
    )
    interaction_cost = get_interaction_cost(
        ego_state_future,
        ado_state_future,
        planner.solver.interaction_cost,
    )
    return interaction_cost.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate prediciton-planning stack using safer_fast and safer_slow scenes"
    )
    parser.add_argument(
        "--load_from",
        type=str,
        required=True,
        help="WandB ID to load trained predictor from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes",
    )
    parser.add_argument(
        "--risk_level",
        type=float,
        nargs="+",
        help="Risk-sensitivity level(s) to test",
        default=[0.95, 1],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        nargs="+",
        help="Number(s) of prediction samples to test",
        default=[1, 4, 16, 64, 256, 1024],
    )
    parser.add_argument(
        "--force_config",
        action="store_true",
        help="""Use this flag to force the use of the local config file
        when loading a model from a checkpoint. Otherwise the checkpoint config file is used.
        In any case the parameters can be overwritten with an argparse argument.""",
    )
    parser.add_argument(
        "--load_last",
        action="store_true",
        help="""Use this flag to force the use of the last checkpoint instead of the best one
        when loading a model.""",
    )

    args = parser.parse_args()
    # Args will be re-parsed, this keeps only the arguments that are compatible with the second parser.
    keep_list = ["--load_from", "--seed", "--load_last", "--force_config"]
    sys.argv = [ss for s in sys.argv for ss in s.split("=")]
    sys.argv = [
        sys.argv[i]
        for i in range(len(sys.argv))
        if sys.argv[i] in keep_list or sys.argv[i - 1] in keep_list or i == 0
    ]
    evaluate_main(
        args.load_from, args.seed, args.num_episodes, args.risk_level, args.num_samples
    )
