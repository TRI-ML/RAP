import os
import io
import pickle
import sys

from functools import partial
from inspect import signature
import matplotlib.pyplot as plt
from tqdm import tqdm

from einops import repeat
import fire
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import torch

from risk_biased.utils.config_argparse import config_argparse
from risk_biased.utils.cost import TTCCostTorch, TTCCostParams, get_cost
from risk_biased.utils.risk import get_risk_estimator

from risk_biased.utils.load_model import load_from_config


def to_device(batch, device):
    output = []
    for item in batch:
        output.append(item.to(device))
    return output


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def distance(pred, truth):
    """
    pred (Tensor): (..., time, xy)
    truth (Tensor): (..., time, xy)
    mask_loss (Tensor): (..., time) Defaults to None.
    """
    return torch.sqrt(torch.sum(torch.square(pred[..., :2] - truth[..., :2]), -1))


def compute_metrics(
    predictor,
    batch,
    cost,
    risk_levels,
    risk_estimator,
    dt,
    unnormalizer,
    n_samples_risk,
    n_samples_stats,
):

    # risk_unbiased
    # risk_biased
    # cost
    # FDE: unbiased, biased(risk_level=[0, 0.3, 0.5, 0.8, 1]) (for all samples so minFDE can be computed later)
    # ADE (for all samples so minADE can be computed later)

    x, mask_x, y, mask_y, mask_loss, map, mask_map, offset, x_ego, y_ego = batch
    mask_z = mask_x.any(-1)

    _, z_mean_inference, z_log_std_inference = predictor.model(
        x,
        mask_x,
        map,
        mask_map,
        offset=offset,
        x_ego=x_ego,
        y_ego=y_ego,
        risk_level=None,
    )

    latent_distribs = {
        "inference": {
            "mean": z_mean_inference[:, 1].detach().cpu(),
            "log_std": z_log_std_inference[:, 1].detach().cpu(),
        }
    }
    inference_distances = []
    cost_list = []
    # Cut the number of samples in packs to avoid out-of-memory problems
    # Compute and store cost for all packs
    for _ in range(n_samples_risk // n_samples_stats):
        z_samples_inference = predictor.model.inference_encoder.sample(
            z_mean_inference,
            z_log_std_inference,
            n_samples=n_samples_stats,
        )

        y_samples = predictor.model.decode(
            z_samples=z_samples_inference,
            mask_z=mask_z,
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            offset=offset,
        )

        mask_loss_samples = repeat(mask_loss, "b a t -> b a s t", s=n_samples_stats)
        # Computing unbiased cost
        cost_list.append(
            get_cost(
                cost,
                x,
                y_samples,
                offset,
                x_ego,
                y_ego,
                dt,
                unnormalizer,
                mask_loss_samples,
            )[:, 1:2]
        )
        inference_distances.append(distance(y_samples, y.unsqueeze(2))[:, 1:2])
    cost_dic = {}
    cost_dic["inference"] = torch.cat(cost_list, 2).detach().cpu()
    distance_dic = {}
    distance_dic["inference"] = torch.cat(inference_distances, 2).detach().cpu()

    # Set up the output risk tensor
    risk_dic = {}

    # Loop on risk_level values to fill the risk estimation for each value and compute stats at each risk level
    for rl in risk_levels:
        risk_level = (
            torch.ones(
                (x.shape[0], x.shape[1]),
                device=x.device,
            )
            * rl
        )
        risk_dic[f"biased_{rl}"] = risk_estimator(
            risk_level[:, 1:2].detach().cpu(), cost_dic["inference"]
        )

        y_samples_biased, z_mean_biased, z_log_std_biased = predictor.model(
            x,
            mask_x,
            map,
            mask_map,
            offset=offset,
            x_ego=x_ego,
            y_ego=y_ego,
            risk_level=risk_level,
            n_samples=n_samples_stats,
        )
        latent_distribs[f"biased_{rl}"] = {
            "mean": z_mean_biased[:, 1].detach().cpu(),
            "log_std": z_log_std_biased[:, 1].detach().cpu(),
        }

        distance_dic[f"biased_{rl}"] = (
            distance(y_samples_biased, y.unsqueeze(2))[:, 1].detach().cpu()
        )
        cost_dic[f"biased_{rl}"] = (
            get_cost(
                cost,
                x,
                y_samples_biased,
                offset,
                x_ego,
                y_ego,
                dt,
                unnormalizer,
                mask_loss_samples,
            )[:, 1]
            .detach()
            .cpu()
        )

    # Return risks for the batch and all risk values
    return {
        "risk": risk_dic,
        "cost": cost_dic,
        "distance": distance_dic,
        "latent_distribs": latent_distribs,
        "mask": mask_loss[:, 1].detach().cpu(),
    }


def cat_metrics_rec(metrics1, metrics2, cat_to):
    for key in metrics1.keys():
        if key not in metrics2.keys():
            raise RuntimeError(
                f"Trying to concatenate objects with different keys: {key} is not in second argument keys."
            )
        elif isinstance(metrics1[key], dict):
            if key not in cat_to.keys():
                cat_to[key] = {}
            cat_metrics_rec(metrics1[key], metrics2[key], cat_to[key])
        elif isinstance(metrics1[key], torch.Tensor):
            cat_to[key] = torch.cat((metrics1[key], metrics2[key]), 0)


def cat_metrics(metrics1, metrics2):
    out = {}
    cat_metrics_rec(metrics1, metrics2, out)
    return out


def masked_mean_std_ste(data, mask):
    mask = mask.view(data.shape)
    norm = mask.sum().clamp_min(1)
    mean = (data * mask).sum() / norm
    std = torch.sqrt(((data - mean) * mask).square().sum() / norm)
    return mean.item(), std.item(), (std / torch.sqrt(norm)).item()


def masked_mean_range(data, mask):
    data = data[mask]
    mean = data.mean()
    min = torch.quantile(data, 0.05)
    max = torch.quantile(data, 0.95)
    return mean, min, max


def masked_mean_dim(data, mask, dim):
    norm = mask.sum(dim).clamp_min(1)
    mean = (data * mask).sum(dim) / norm
    return mean


def plot_risk_error(metrics, risk_levels, risk_estimator, max_n_samples, path_save):
    cost_inference = metrics["cost"]["inference"]
    cost_biased_0 = metrics["cost"]["biased_0"]
    mask = metrics["mask"].any(1)
    ones_tensor = torch.ones(mask.shape[0])
    n_samples = np.minimum(cost_biased_0.shape[1], max_n_samples)

    for rl in risk_levels:
        key = f"biased_{rl}"
        reference_risk = metrics["risk"][key]
        mean_inference_risk_error_per_samples = np.zeros(n_samples - 1)
        min_inference_risk_error_per_samples = np.zeros(n_samples - 1)
        max_inference_risk_error_per_samples = np.zeros(n_samples - 1)
        # mean_biased_0_risk_error_per_samples = np.zeros(n_samples-1)
        # min_biased_0_risk_error_per_samples = np.zeros(n_samples-1)
        # max_biased_0_risk_error_per_samples = np.zeros(n_samples-1)
        mean_biased_risk_error_per_samples = np.zeros(n_samples - 1)
        min_biased_risk_error_per_samples = np.zeros(n_samples - 1)
        max_biased_risk_error_per_samples = np.zeros(n_samples - 1)
        risk_level_tensor = ones_tensor * rl
        for sub_samples in range(1, n_samples):
            perm = torch.randperm(metrics["cost"][key].shape[1])[:sub_samples]
            risk_error_biased = metrics["cost"][key][:, perm].mean(1) - reference_risk
            (
                mean_biased_risk_error_per_samples[sub_samples - 1],
                min_biased_risk_error_per_samples[sub_samples - 1],
                max_biased_risk_error_per_samples[sub_samples - 1],
            ) = masked_mean_range(risk_error_biased, mask)
            risk_error_inference = (
                risk_estimator(risk_level_tensor, cost_inference[:, :, :sub_samples])
                - reference_risk
            )
            (
                mean_inference_risk_error_per_samples[sub_samples - 1],
                min_inference_risk_error_per_samples[sub_samples - 1],
                max_inference_risk_error_per_samples[sub_samples - 1],
            ) = masked_mean_range(risk_error_inference, mask)
            # risk_error_biased_0 = risk_estimator(risk_level_tensor, cost_biased_0[:, :sub_samples]) - reference_risk
            # (mean_biased_0_risk_error_per_samples[sub_samples-1], min_biased_0_risk_error_per_samples[sub_samples-1], max_biased_0_risk_error_per_samples[sub_samples-1]) = masked_mean_range(risk_error_biased_0, mask)

        plt.plot(
            range(1, n_samples),
            mean_inference_risk_error_per_samples,
            label="Inference",
        )
        plt.fill_between(
            range(1, n_samples),
            min_inference_risk_error_per_samples,
            max_inference_risk_error_per_samples,
            alpha=0.3,
        )

        # plt.plot(range(1, n_samples), mean_biased_0_risk_error_per_samples, label="Unbiased")
        # plt.fill_between(range(1, n_samples), min_biased_0_risk_error_per_samples, max_biased_0_risk_error_per_samples, alpha=.3)

        plt.plot(
            range(1, n_samples), mean_biased_risk_error_per_samples, label="Biased"
        )
        plt.fill_between(
            range(1, n_samples),
            min_biased_risk_error_per_samples,
            max_biased_risk_error_per_samples,
            alpha=0.3,
        )
        plt.ylim(
            np.min(min_inference_risk_error_per_samples),
            np.max(max_biased_risk_error_per_samples),
        )

        plt.hlines(y=0, xmin=0, xmax=n_samples, colors="black", linestyles="--", lw=0.3)

        plt.xlabel("Number of samples")
        plt.ylabel("Risk estimation error")
        plt.legend()
        plt.title(f"Risk estimation error at risk-level={rl}")
        plt.gcf().set_size_inches(4, 3)
        plt.legend(loc="lower right")
        plt.savefig(fname=os.path.join(path_save, f"risk_level_{rl}.svg"))
        plt.savefig(fname=os.path.join(path_save, f"risk_level_{rl}.png"))
        plt.clf()
        # plt.show()


def compute_stats(metrics, n_samples_mean_cost=4):
    biased_risk_estimate = {}
    for key in metrics["cost"].keys():
        if key == "inference":
            continue
        risk = metrics["risk"][key]
        mean_cost = metrics["cost"][key][:, :n_samples_mean_cost].mean(1)
        risk_error = mean_cost - risk
        biased_risk_estimate[key] = {}
        (
            biased_risk_estimate[key]["mean"],
            biased_risk_estimate[key]["std"],
            biased_risk_estimate[key]["ste"],
        ) = masked_mean_std_ste(risk_error, metrics["mask"].any(1))

        (
            biased_risk_estimate[key]["mean_abs"],
            biased_risk_estimate[key]["std_abs"],
            biased_risk_estimate[key]["ste_abs"],
        ) = masked_mean_std_ste(risk_error.abs(), metrics["mask"].any(1))

    risk_stats = {}
    for key in metrics["risk"].keys():
        risk_stats[key] = {}
        (
            risk_stats[key]["mean"],
            risk_stats[key]["std"],
            risk_stats[key]["ste"],
        ) = masked_mean_std_ste(metrics["risk"][key], metrics["mask"].any(1))

    cost_stats = {}
    for key in metrics["cost"].keys():
        cost_stats[key] = {}
        (
            cost_stats[key]["mean"],
            cost_stats[key]["std"],
            cost_stats[key]["ste"],
        ) = masked_mean_std_ste(
            metrics["cost"][key], metrics["mask"].any(-1, keepdim=True)
        )

    distance_stats = {}
    for key in metrics["distance"].keys():
        distance_stats[key] = {"FDE": {}, "ADE": {}, "minFDE": {}, "minADE": {}}
        (
            distance_stats[key]["FDE"]["mean"],
            distance_stats[key]["FDE"]["std"],
            distance_stats[key]["FDE"]["ste"],
        ) = masked_mean_std_ste(
            metrics["distance"][key][..., -1], metrics["mask"][:, None, -1]
        )
        mean_dist = masked_mean_dim(
            metrics["distance"][key], metrics["mask"][:, None, :], -1
        )
        (
            distance_stats[key]["ADE"]["mean"],
            distance_stats[key]["ADE"]["std"],
            distance_stats[key]["ADE"]["ste"],
        ) = masked_mean_std_ste(mean_dist, metrics["mask"].any(-1, keepdim=True))
        for i in [6, 16, 32]:
            distance_stats[key]["minFDE"][i] = {}
            min_dist, _ = metrics["distance"][key][:, :i, -1].min(1)
            (
                distance_stats[key]["minFDE"][i]["mean"],
                distance_stats[key]["minFDE"][i]["std"],
                distance_stats[key]["minFDE"][i]["ste"],
            ) = masked_mean_std_ste(min_dist, metrics["mask"][:, -1])
            distance_stats[key]["minADE"][i] = {}
            mean_dist, _ = masked_mean_dim(
                metrics["distance"][key][:, :i], metrics["mask"][:, None, :], -1
            ).min(1)
            (
                distance_stats[key]["minADE"][i]["mean"],
                distance_stats[key]["minADE"][i]["std"],
                distance_stats[key]["minADE"][i]["ste"],
            ) = masked_mean_std_ste(mean_dist, metrics["mask"].any(-1))
    return {
        "risk": risk_stats,
        "biased_risk_estimate": biased_risk_estimate,
        "cost": cost_stats,
        "distance": distance_stats,
    }


def print_stats(stats, n_samples_mean_cost=4):
    slash = "\\"
    brace_open = "{"
    brace_close = "}"
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print(
        f"Predictive Model & ${slash}sigma$ & minFDE(16) & FDE (1) & Risk est. error ({n_samples_mean_cost}) & Risk est. $|$error$|$ ({n_samples_mean_cost}) {slash}{slash}"
    )
    print("\\hline")

    for key in stats["distance"].keys():
        strg = (
            f"  ${stats['distance'][key]['minFDE'][16]['mean']:.2f}$ {slash}scriptsize{brace_open}${slash}pm {stats['distance'][key]['minFDE'][16]['ste']:.2f}${brace_close}"
            + f"& ${stats['distance'][key]['FDE']['mean']:.2f}$ {slash}scriptsize{brace_open}${slash}pm {stats['distance'][key]['FDE']['ste']:.2f}${brace_close}"
        )

        if key == "inference":
            strg = (
                "Unbiased CVAE & "
                + f"{slash}scriptsize{brace_open}NA{brace_close} &"
                + strg
                + f"& {slash}scriptsize{brace_open}NA{brace_close} & {slash}scriptsize{brace_open}NA{brace_close} {slash}{slash}"
            )
            print(strg)
            print("\\hline")
        else:
            strg = (
                "Biased CVAE & "
                + f"{key[7:]} & "
                + strg
                + f"& ${stats['biased_risk_estimate'][key]['mean']:.2f}$ {slash}scriptsize{brace_open}${slash}pm {stats['biased_risk_estimate'][key]['ste']:.2f}${brace_close}"
                + f"& ${stats['biased_risk_estimate'][key]['mean_abs']:.2f}$ {slash}scriptsize{brace_open}${slash}pm {stats['biased_risk_estimate'][key]['ste_abs']:.2f}${brace_close}"
                + f"{slash}{slash}"
            )
            print(strg)
    print("\\hline")
    print("\\end{tabular}")


def main(
    log_path,
    force_recompute,
    n_samples_risk=256,
    n_samples_stats=32,
    n_samples_plot=16,
    args_to_parser=[],
):
    # Overwrite sys.argv so it doesn't mess up the parser.
    sys.argv = sys.argv[0:1] + args_to_parser
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    waymo_config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "waymo_config.py"
    )
    cfg = config_argparse([config_path, waymo_config_path])

    file_path = os.path.join(log_path, f"metrics_{cfg.load_from}.pickle")
    fig_path = os.path.join(log_path, f"plots_{cfg.load_from}")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    risk_levels = [0, 0.3, 0.5, 0.8, 0.95, 1]
    cost = TTCCostTorch(TTCCostParams.from_config(cfg))
    risk_estimator = get_risk_estimator(cfg.risk_estimator)
    n_samples_mean_cost = 4

    if not os.path.exists(file_path) or force_recompute:
        with torch.no_grad():
            if cfg.seed is not None:
                seed_everything(cfg.seed)

            predictor, dataloaders, cfg = load_from_config(cfg)
            device = torch.device(cfg.gpus[0])
            predictor = predictor.to(device)

            val_loader = dataloaders.val_dataloader(shuffle=False, drop_last=False)

            # This loops over batches in the validation dataset
            beg = 0
            metrics_all = None
            for val_batch in tqdm(val_loader):
                end = beg + val_batch[0].shape[0]
                metrics = compute_metrics(
                    predictor=predictor,
                    batch=to_device(val_batch, device),
                    cost=cost,
                    risk_levels=risk_levels,
                    risk_estimator=risk_estimator,
                    dt=cfg.dt,
                    unnormalizer=dataloaders.unnormalize_trajectory,
                    n_samples_risk=n_samples_risk,
                    n_samples_stats=n_samples_stats,
                )
                if metrics_all is None:
                    metrics_all = metrics
                else:
                    metrics_all = cat_metrics(metrics_all, metrics)
                beg = end
        with open(file_path, "wb") as handle:
            pickle.dump(metrics_all, handle)
    else:
        print(f"Loading pre-computed metrics from {file_path}")
        with open(file_path, "rb") as handle:
            metrics_all = CPU_Unpickler(handle).load()

    stats = compute_stats(metrics_all, n_samples_mean_cost=n_samples_mean_cost)
    print_stats(stats, n_samples_mean_cost=n_samples_mean_cost)
    plot_risk_error(metrics_all, risk_levels, risk_estimator, n_samples_plot, fig_path)


if __name__ == "__main__":
    # main("./logs/002/", False, 256, 32, 16)
    # Fire turns the main function into a script, then the risk_biased module argparse reads the other arguments.
    # Thus, the way to use it would be:
    #     >python compute_stats.py <path to existing log dir> <Force recompute> <n_samples_risk> <n_samples_stats> <n_samples_plot> <other argparse arguments, example --load_from 1uail32>

    # This is a hack to separate the Fire script args from the argparse arguments
    args_to_parser = sys.argv[len(signature(main).parameters) :]
    partial_main = partial(main, args_to_parser=args_to_parser)
    sys.argv = sys.argv[: len(signature(main).parameters)]

    # Runs the main as a script
    fire.Fire(partial_main)
