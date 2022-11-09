import os

from mmcv import Config
import wandb

from risk_biased.utils.config_argparse import config_argparse


def get_config(log_dir: str, is_interaction: bool = False) -> Config:
    wandb.login()
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "risk_biased", "config", "learning_config.py"
    )
    if is_interaction:
        waymo_config_path = os.path.join(
            working_dir, "..", "..", "risk_biased", "config", "waymo_config.py"
        )
        cfg = config_argparse([config_path, waymo_config_path])
    else:
        cfg = config_argparse(config_path)

    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        dir=log_dir,
        resume="allow",
        config=dict(cfg),
    )

    # Allow WandB to update the config
    cfg.update(wandb.config)

    return cfg
