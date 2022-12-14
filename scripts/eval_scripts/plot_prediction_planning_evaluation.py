"""plot_prediction_planning_evaluation.py --load_from <wandb ID> --seed <seed>
--scene_type <safer_fast or safer_slow> --risk_level <a list of risk-levels>
--num_samples <a list of numbers of prediction samples>

This script plots statistics of evaluation results generated by
`evaluate_prediction_planning_stack.py`.
"""


import argparse
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def plot_main(
    stats_dir: str,
    scene_type: str,
    risk_level_list: List[float],
    num_prediction_samples_list: List[int],
) -> None:
    if 0.0 in risk_level_list:
        plot_computation_time(
            stats_dir,
            scene_type,
            num_prediction_samples_list=num_prediction_samples_list,
        )
    plot_varying_risk(
        stats_dir,
        scene_type,
        num_prediction_samples_list=[num_prediction_samples_list[-1]],
        risk_level_list=risk_level_list,
        risk_in_planner=True,
    )
    plot_varying_risk(
        stats_dir,
        scene_type,
        num_prediction_samples_list=[num_prediction_samples_list[-1]],
        risk_level_list=risk_level_list,
        risk_in_planner=False,
    )
    plot_policy_comparison(
        stats_dir,
        scene_type,
        num_prediction_samples_list=num_prediction_samples_list,
        risk_level_list=list(filter(lambda r: r != 0.0, risk_level_list)),
    )


# How does computation time scale as we increase the number of samples?
def plot_computation_time(
    stats_dir: str,
    scene_type: str,
    num_prediction_samples_list: List[int],
    alpha_for_confint: float = 0.95,
) -> None:
    risk_level = 0.0
    stats_dict_zero_risk = dict()
    computation_time_mean_list, computation_time_sem_list = [], []
    for num_samples in num_prediction_samples_list:
        file_path = os.path.join(
            stats_dir,
            f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}.pkl",
        )
        assert os.path.exists(
            file_path
        ), f"missing experiment with num_samples == {num_samples} and risk_level == {risk_level}"
        with open(file_path, "rb") as f:
            stats_dict_zero_risk[num_samples] = pickle.load(f)

        num_episodes = _get_num_episodes(stats_dict_zero_risk[num_samples])
        computation_time_list = [
            stats_dict_zero_risk[num_samples][idx]["computation_time_ms"]
            for idx in range(num_episodes)
        ]
        computation_time_mean_list.append(np.mean(computation_time_list))
        computation_time_sem_list.append(st.sem(computation_time_list))

    # ref: https://www.statology.org/confidence-intervals-python/
    confint_lower, confint_upper = st.norm.interval(
        alpha=alpha_for_confint,
        loc=computation_time_mean_list,
        scale=computation_time_sem_list,
    )

    _, ax = plt.subplots(1, figsize=(6, 6))

    ax.plot(
        num_prediction_samples_list,
        computation_time_mean_list,
        color="skyblue",
        linewidth=2.0,
    )
    ax.fill_between(
        num_prediction_samples_list,
        confint_upper,
        confint_lower,
        facecolor="skyblue",
        alpha=0.3,
    )
    ax.set_xlabel("Number of Prediction Samples")
    ax.set_ylabel("Computation Time for Prediction and Planning (ms)")

    plt.show()


# How do varying risk-levels affect the safety/efficiency of the policy?
def plot_varying_risk(
    stats_dir: str,
    scene_type: str,
    num_prediction_samples_list: List[int],
    risk_level_list: List[float],
    risk_in_planner: bool = False,
    alpha_for_confint: float = 0.95,
) -> None:
    _, ax = plt.subplots(
        1,
        len(num_prediction_samples_list),
        figsize=(6 * len(num_prediction_samples_list), 6),
    )
    if not type(ax) == np.ndarray:
        ax = [ax]
    stats_dict = dict()
    suptitle = "Safety-Efficiency Tradeoff of Optimized Policy"
    if risk_in_planner:
        suptitle += " (Risk in Planner)"
    else:
        suptitle += " (Risk in Predictor)"
    plt.suptitle(suptitle)
    for (plot_idx, num_samples) in enumerate(num_prediction_samples_list):
        stats_dict[num_samples] = dict()
        interaction_cost_mean_list, interaction_cost_sem_list = [], []
        tracking_cost_mean_list, tracking_cost_sem_list = [], []
        for risk_level in risk_level_list:
            if risk_level == 0.0:
                file_path = os.path.join(
                    stats_dir,
                    f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}.pkl",
                )
            elif risk_in_planner:
                file_path = os.path.join(
                    stats_dir,
                    f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}_in_planner.pkl",
                )
            else:
                file_path = os.path.join(
                    stats_dir,
                    f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}_in_predictor.pkl",
                )
            assert os.path.exists(
                file_path
            ), f"missing experiment with num_samples == {num_samples} and risk_level == {risk_level}"

            with open(file_path, "rb") as f:
                stats_dict[num_samples][risk_level] = pickle.load(f)
            num_episodes = _get_num_episodes(stats_dict[num_samples][risk_level])

            interaction_cost_list = [
                stats_dict[num_samples][risk_level][idx][
                    "interaction_cost_ground_truth"
                ]
                for idx in range(num_episodes)
            ]
            interaction_cost_mean_list.append(np.mean(interaction_cost_list))
            interaction_cost_sem_list.append(st.sem(interaction_cost_list))

            tracking_cost_list = [
                stats_dict[num_samples][risk_level][idx]["tracking_cost"]
                for idx in range(num_episodes)
            ]
            tracking_cost_mean_list.append(np.mean(tracking_cost_list))
            tracking_cost_sem_list.append(st.sem(tracking_cost_list))

        (
            interaction_cost_confint_lower,
            interaction_cost_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=interaction_cost_mean_list,
            scale=interaction_cost_sem_list,
        )

        (tracking_cost_confint_lower, tracking_cost_confint_upper,) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=tracking_cost_mean_list,
            scale=tracking_cost_sem_list,
        )

        ax[plot_idx].plot(
            risk_level_list,
            interaction_cost_mean_list,
            color="orange",
            linewidth=2.0,
            label="ground-truth collision cost",
        )
        ax[plot_idx].fill_between(
            risk_level_list,
            interaction_cost_confint_upper,
            interaction_cost_confint_lower,
            color="orange",
            alpha=0.3,
        )

        ax[plot_idx].plot(
            risk_level_list,
            tracking_cost_mean_list,
            color="lightgreen",
            linewidth=2.0,
            label="trajectory tracking cost",
        )
        ax[plot_idx].fill_between(
            risk_level_list,
            tracking_cost_confint_upper,
            tracking_cost_confint_lower,
            color="lightgreen",
            alpha=0.3,
        )

        if risk_in_planner:
            ax[plot_idx].set_xlabel("Risk-Sensitivity Level (in Planner)")
        else:
            ax[plot_idx].set_xlabel("Risk-Sensitivity Level (in Predictor)")
        ax[plot_idx].set_ylabel("Cost")
        ax[plot_idx].set_title(f"Number of Prediction Samples: {num_samples}")
        ax[plot_idx].legend(loc="upper right")

    plt.show()


# How does (risk-biased predictor + risk-neutral planner) compare with (risk-neutral predictor + risk-sensitive planner)
# in terms of characteristics of the optimized policy?
def plot_policy_comparison(
    stats_dir: str,
    scene_type: str,
    num_prediction_samples_list: List[int],
    risk_level_list: List[float],
    alpha_for_confint: float = 0.95,
) -> None:
    assert not 0.0 in risk_level_list
    num_rows = 4
    _, ax = plt.subplots(
        num_rows, len(risk_level_list), figsize=(6 * len(risk_level_list), 6 * num_rows)
    )
    if len(risk_level_list) == 1:
        for row_idx in range(num_rows):
            ax[row_idx] = [ax[row_idx]]
    suptitle = "Characteristics of Optimized Policy"
    plt.suptitle(suptitle)
    predictor_stats_dict, planner_stats_dict = dict(), dict()
    for (plot_idx, risk_level) in enumerate(risk_level_list):
        predictor_stats_dict[risk_level], planner_stats_dict[risk_level] = (
            dict(),
            dict(),
        )
        predictor_interaction_cost_mean_list, planner_interaction_cost_mean_list = (
            [],
            [],
        )
        predictor_interaction_cost_sem_list, planner_interaction_cost_sem_list = [], []
        predictor_tracking_cost_mean_list, planner_tracking_cost_mean_list = [], []
        predictor_tracking_cost_sem_list, planner_tracking_cost_sem_list = [], []
        predictor_interaction_risk_mean_list, planner_interaction_risk_mean_list = (
            [],
            [],
        )
        predictor_interaction_risk_sem_list, planner_interaction_risk_sem_list = (
            [],
            [],
        )
        predictor_total_objective_mean_list, planner_total_objective_mean_list = (
            [],
            [],
        )
        predictor_total_objective_sem_list, planner_total_objective_sem_list = (
            [],
            [],
        )
        for num_samples in num_prediction_samples_list:
            file_path = os.path.join(
                stats_dir,
                f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}_in_predictor.pkl",
            )
            assert os.path.exists(
                file_path
            ), f"missing experiment with num_samples == {num_samples} and risk_level == {risk_level}"
            with open(file_path, "rb") as f:
                predictor_stats_dict[risk_level][num_samples] = pickle.load(f)
            predictor_num_episodes = _get_num_episodes(
                predictor_stats_dict[risk_level][num_samples]
            )
            predictor_interaction_cost_list = [
                predictor_stats_dict[risk_level][num_samples][idx][
                    "interaction_cost_ground_truth"
                ]
                for idx in range(predictor_num_episodes)
            ]
            predictor_interaction_cost_mean_list.append(
                np.mean(predictor_interaction_cost_list)
            )
            predictor_interaction_cost_sem_list.append(
                st.sem(predictor_interaction_cost_list)
            )
            predictor_tracking_cost_list = [
                predictor_stats_dict[risk_level][num_samples][idx]["tracking_cost"]
                for idx in range(predictor_num_episodes)
            ]
            predictor_tracking_cost_mean_list.append(
                np.mean(predictor_tracking_cost_list)
            )
            predictor_tracking_cost_sem_list.append(
                st.sem(predictor_tracking_cost_list)
            )
            predictor_interaction_risk_list = [
                predictor_stats_dict[risk_level][num_samples][idx]["interaction_risk"]
                for idx in range(predictor_num_episodes)
            ]
            predictor_interaction_risk_mean_list.append(
                np.mean(predictor_interaction_risk_list)
            )
            predictor_interaction_risk_sem_list.append(
                st.sem(predictor_interaction_risk_list)
            )
            predictor_total_objective_list = [
                interaction_risk + tracking_cost
                for (interaction_risk, tracking_cost) in zip(
                    predictor_interaction_risk_list, predictor_tracking_cost_list
                )
            ]
            predictor_total_objective_mean_list.append(
                np.mean(predictor_total_objective_list)
            )
            predictor_total_objective_sem_list.append(
                st.sem(predictor_total_objective_list)
            )

            file_path = os.path.join(
                stats_dir,
                f"{scene_type}_{num_samples}_samples_risk_level_{risk_level}_in_planner.pkl",
            )
            assert os.path.exists(
                file_path
            ), f"missing experiment with num_samples == {num_samples} and risk_level == {risk_level}"
            with open(file_path, "rb") as f:
                planner_stats_dict[risk_level][num_samples] = pickle.load(f)
            planner_num_episodes = _get_num_episodes(
                planner_stats_dict[risk_level][num_samples]
            )
            planner_interaction_cost_list = [
                planner_stats_dict[risk_level][num_samples][idx][
                    "interaction_cost_ground_truth"
                ]
                for idx in range(planner_num_episodes)
            ]
            planner_interaction_cost_mean_list.append(
                np.mean(planner_interaction_cost_list)
            )
            planner_interaction_cost_sem_list.append(
                st.sem(planner_interaction_cost_list)
            )
            planner_tracking_cost_list = [
                planner_stats_dict[risk_level][num_samples][idx]["tracking_cost"]
                for idx in range(planner_num_episodes)
            ]
            planner_tracking_cost_mean_list.append(np.mean(planner_tracking_cost_list))
            planner_tracking_cost_sem_list.append(st.sem(planner_tracking_cost_list))
            planner_interaction_risk_list = [
                planner_stats_dict[risk_level][num_samples][idx]["interaction_risk"]
                for idx in range(planner_num_episodes)
            ]
            planner_interaction_risk_mean_list.append(
                np.mean(planner_interaction_risk_list)
            )
            planner_interaction_risk_sem_list.append(
                st.sem(planner_interaction_risk_list)
            )
            planner_total_objective_list = [
                interaction_risk + tracking_cost
                for (interaction_risk, tracking_cost) in zip(
                    planner_interaction_risk_list, planner_tracking_cost_list
                )
            ]
            planner_total_objective_mean_list.append(
                np.mean(planner_total_objective_list)
            )
            planner_total_objective_sem_list.append(
                st.sem(planner_total_objective_list)
            )

        (
            predictor_interaction_cost_confint_lower,
            predictor_interaction_cost_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=predictor_interaction_cost_mean_list,
            scale=predictor_interaction_cost_sem_list,
        )
        (
            predictor_tracking_cost_confint_lower,
            predictor_tracking_cost_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=predictor_tracking_cost_mean_list,
            scale=predictor_tracking_cost_sem_list,
        )
        (
            predictor_interaction_risk_confint_lower,
            predictor_interaction_risk_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=predictor_interaction_risk_mean_list,
            scale=predictor_interaction_risk_sem_list,
        )
        (
            predictor_total_objective_confint_lower,
            predictor_total_objective_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=predictor_total_objective_mean_list,
            scale=predictor_total_objective_sem_list,
        )

        (
            planner_interaction_cost_confint_lower,
            planner_interaction_cost_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=planner_interaction_cost_mean_list,
            scale=planner_interaction_cost_sem_list,
        )
        (
            planner_tracking_cost_confint_lower,
            planner_tracking_cost_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=planner_tracking_cost_mean_list,
            scale=planner_tracking_cost_sem_list,
        )
        (
            planner_interaction_risk_confint_lower,
            planner_interaction_risk_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=planner_interaction_risk_mean_list,
            scale=planner_interaction_risk_sem_list,
        )
        (
            planner_total_objective_confint_lower,
            planner_total_objective_confint_upper,
        ) = st.norm.interval(
            alpha=alpha_for_confint,
            loc=planner_total_objective_mean_list,
            scale=planner_total_objective_sem_list,
        )

        ax[0][plot_idx].plot(
            num_prediction_samples_list,
            planner_interaction_cost_mean_list,
            color="skyblue",
            linewidth=2.0,
            label="risk in planner",
        )
        ax[0][plot_idx].fill_between(
            num_prediction_samples_list,
            planner_interaction_cost_confint_upper,
            planner_interaction_cost_confint_lower,
            color="skyblue",
            alpha=0.3,
        )
        ax[0][plot_idx].plot(
            num_prediction_samples_list,
            predictor_interaction_cost_mean_list,
            color="orange",
            linewidth=2.0,
            label="risk in predictor",
        )
        ax[0][plot_idx].fill_between(
            num_prediction_samples_list,
            predictor_interaction_cost_confint_upper,
            predictor_interaction_cost_confint_lower,
            color="orange",
            alpha=0.3,
        )
        ax[0][plot_idx].set_xlabel("Number of Prediction Samples")
        ax[0][plot_idx].set_ylabel("Ground-Truth Collision Cost")
        ax[0][plot_idx].set_title(f"Risk-Sensitivity Level: {risk_level}")
        ax[0][plot_idx].legend(loc="upper right")
        ax[0][plot_idx].set_xscale("log")

        ax[1][plot_idx].plot(
            num_prediction_samples_list,
            planner_tracking_cost_mean_list,
            color="skyblue",
            linewidth=2.0,
            label="risk in planner",
        )
        ax[1][plot_idx].fill_between(
            num_prediction_samples_list,
            planner_tracking_cost_confint_upper,
            planner_tracking_cost_confint_lower,
            color="skyblue",
            alpha=0.3,
        )
        ax[1][plot_idx].plot(
            num_prediction_samples_list,
            predictor_tracking_cost_mean_list,
            color="orange",
            linewidth=2.0,
            label="risk in predictor",
        )
        ax[1][plot_idx].fill_between(
            num_prediction_samples_list,
            predictor_tracking_cost_confint_upper,
            predictor_tracking_cost_confint_lower,
            color="orange",
            alpha=0.3,
        )
        ax[1][plot_idx].set_xlabel("Number of Prediction Samples")
        ax[1][plot_idx].set_ylabel("Trajectory Tracking Cost")
        # ax[1][plot_idx].set_title(f"Risk-Sensitivity Level: {risk_level}")
        ax[1][plot_idx].legend(loc="lower right")
        ax[1][plot_idx].set_xscale("log")

        ax[2][plot_idx].plot(
            num_prediction_samples_list,
            planner_interaction_risk_mean_list,
            color="skyblue",
            linewidth=2.0,
            label="risk in planner",
        )
        ax[2][plot_idx].fill_between(
            num_prediction_samples_list,
            planner_interaction_risk_confint_upper,
            planner_interaction_risk_confint_lower,
            color="skyblue",
            alpha=0.3,
        )
        ax[2][plot_idx].plot(
            num_prediction_samples_list,
            predictor_interaction_risk_mean_list,
            color="orange",
            linewidth=2.0,
            label="risk in predictor",
        )
        ax[2][plot_idx].fill_between(
            num_prediction_samples_list,
            predictor_interaction_risk_confint_upper,
            predictor_interaction_risk_confint_lower,
            color="orange",
            alpha=0.3,
        )
        ax[2][plot_idx].set_xlabel("Number of Prediction Samples")
        ax[2][plot_idx].set_ylabel("Collision Risk")
        # ax[2][plot_idx].set_title(f"Risk-Sensitivity Level: {risk_level}")
        ax[2][plot_idx].legend(loc="upper right")
        ax[2][plot_idx].set_xscale("log")

        ax[3][plot_idx].plot(
            num_prediction_samples_list,
            planner_total_objective_mean_list,
            color="skyblue",
            linewidth=2.0,
            label="risk in planner",
        )
        ax[3][plot_idx].fill_between(
            num_prediction_samples_list,
            planner_total_objective_confint_upper,
            planner_total_objective_confint_lower,
            color="skyblue",
            alpha=0.3,
        )
        ax[3][plot_idx].plot(
            num_prediction_samples_list,
            predictor_total_objective_mean_list,
            color="orange",
            linewidth=2.0,
            label="risk in predictor",
        )
        ax[3][plot_idx].fill_between(
            num_prediction_samples_list,
            predictor_total_objective_confint_upper,
            predictor_total_objective_confint_lower,
            color="orange",
            alpha=0.3,
        )
        ax[3][plot_idx].set_xlabel("Number of Prediction Samples")
        ax[3][plot_idx].set_ylabel("Planner's Total Objective")
        # ax[3][plot_idx].set_title(f"Risk-Sensitivity Level: {risk_level}")
        ax[3][plot_idx].legend(loc="upper right")
        ax[3][plot_idx].set_xscale("log")

    plt.show()


def _get_num_episodes(stats_dict: dict):
    return max(filter(lambda key: type(key) == int, stats_dict)) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="visualize evaluation result of evaluate_prediction_planning_stack.py"
    )
    parser.add_argument(
        "--load_from",
        type=str,
        required=True,
        help="WandB ID for specification of trained predictor",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--scene_type",
        type=str,
        choices=["safer_fast", "safer_slow"],
        required=True,
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
    args = parser.parse_args()
    dir_name = "planner_eval"
    stats_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "logs",
        dir_name,
        f"run-{args.load_from}_{args.seed}",
    )
    postfix_string = ""
    assert os.path.exists(
        stats_dir
    ), f"{stats_dir} does not exist. Did you run 'evaluate_prediction_planning_stack{postfix_string}.py --load_from {args.load_from} --seed {args.seed}' ?"

    plot_main(stats_dir, args.scene_type, args.risk_level, args.num_samples)
