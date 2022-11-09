import os
import shutil

from mmcv import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

import wandb

from risk_biased.utils.callbacks import SwitchTrainingModeCallback
from risk_biased.utils.callbacks import (
    HistogramCallback,
    PlotTrajCallback,
    DrawCallbackParams,
)
from risk_biased.utils.load_model import load_from_config
from scripts.scripts_utils.load_utils import get_config


def create_log_dir():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(working_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


def save_log_config(cfg: Config, predictor):
    # Save and log the config (not only a copy of the config file because settings may have been overwritten by argparse)
    log_config_path = os.path.join(wandb.run.dir, "learning_config.py")
    cfg.dump(log_config_path)
    wandb.save(log_config_path)
    # Save files listed in the current wandb log dir
    for file_name in cfg.files_to_log:
        dest_path = os.path.join(wandb.run.dir, os.path.basename(file_name))
        shutil.copy(file_name, dest_path)
        wandb.save(dest_path)

    if cfg.log_weights_and_grads:
        wandb.watch(predictor, log="all", log_freq=100)


def create_callbacks(cfg: Config, log_dir: str, is_interaction: bool) -> list:
    # Save checkpoint of last model in a specific directory
    last_run_checkpoint_callback = ModelCheckpoint(
        monitor="val/minfde/prior",
        mode="min",
        filename="epoch={epoch:02d}-step={step}-val_minfde_prior={val/minfde/prior:.2f}",
        auto_insert_metric_name=False,
        dirpath=os.path.join(log_dir, "checkpoints_last_run"),
        save_last=True,
    )

    # Save checkpoints of current run in current wandb log dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val/minfde/prior",
        mode="min",
        filename="epoch={epoch:02d}-step={step}-val_minfde_prior={val/minfde/prior:.2f}",
        auto_insert_metric_name=False,
        dirpath=wandb.run.dir,
        save_last=True,
    )
    callbacks = [
        last_run_checkpoint_callback,
        checkpoint_callback,
    ]

    if not is_interaction:
        histogram_callback = HistogramCallback(
            params=DrawCallbackParams.from_config(cfg),
            n_samples=1000,
        )

        plot_callback = PlotTrajCallback(
            params=DrawCallbackParams.from_config(cfg), n_samples=10
        )
        callbacks.append(histogram_callback)
        callbacks.append(plot_callback)

    if cfg.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val/minfde/prior",
            min_delta=-0.2,
            patience=5,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stopping_callback)

    switch_mode_callback = SwitchTrainingModeCallback(
        switch_at_epoch=cfg.num_epochs_cvae
    )
    callbacks.append(switch_mode_callback)

    return callbacks


def get_trainer(cfg: Config, logger: WandbLogger, callbacks: list) -> Trainer:

    num_epochs = cfg.num_epochs_cvae + cfg.num_epochs_bias

    return Trainer(
        gpus=cfg.gpus,
        max_epochs=num_epochs,
        logger=logger,
        val_check_interval=float(cfg.val_check_interval_epoch),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=callbacks,
    )


def main(is_interaction: bool = False):

    log_dir = create_log_dir()
    cfg = get_config(log_dir, is_interaction)

    predictor, dataloaders, cfg = load_from_config(cfg)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    save_log_config(cfg, predictor)

    logger = WandbLogger(
        project=cfg.project, log_model=True, save_dir=log_dir, id=wandb.run.id
    )

    callbacks = create_callbacks(cfg, log_dir, is_interaction)

    trainer = get_trainer(cfg, logger, callbacks)

    trainer.fit(
        predictor,
        train_dataloaders=dataloaders.train_dataloader(),
        val_dataloaders=dataloaders.val_dataloader(),
    )

    wandb.finish()


if __name__ == "__main__":
    main(is_interaction=True)
