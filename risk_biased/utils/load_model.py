from typing import Callable
import os
from typing import Optional, Tuple, Union
import warnings

from mmcv import Config
import torch
import wandb

from risk_biased.predictors.biased_predictor import (
    LitTrajectoryPredictor,
    LitTrajectoryPredictorParams,
)

from risk_biased.utils.config_argparse import config_argparse
from risk_biased.utils.cost import TTCCostParams
from risk_biased.utils.torch_utils import load_weights

from risk_biased.scene_dataset.loaders import SceneDataLoaders
from risk_biased.scene_dataset.scene import load_create_dataset

from risk_biased.utils.waymo_dataloader import WaymoDataloaders


def get_predictor(
    config: Config, unnormalizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    params = LitTrajectoryPredictorParams.from_config(config)
    model_class = LitTrajectoryPredictor
    ttc_params = TTCCostParams.from_config(config)
    return model_class(params=params, unnormalizer=unnormalizer, cost_params=ttc_params)


def load_from_wandb_id(
    log_id: str,
    log_path: str,
    entity: str,
    project: str,
    config: Optional[Config] = None,
    load_last=False,
) -> Tuple[Union[LitTrajectoryPredictor, LitTrajectoryPredictor], Config]:
    """
    Load a model using a wandb id code.
    Args:
        log_id: the wandb id code
        log_path: the wandb log directory path
        config: An optional configuration argument, use these settings if not None, use the settings from the log directory otherwise
        load_last: An optional argumument, set to True to load the last checkpoint instead of the best one
    Returns:
        Predictor model and config file either loaded from the checkpoint or the one passed as argument.
    """
    list_matching = list(filter(lambda path: log_id in path, os.listdir(log_path)))
    if len(list_matching) == 1:
        list_ckpt = list(
            filter(
                lambda path: "epoch" in path and ".ckpt" in path,
                os.listdir(os.path.join(log_path, list_matching[0], "files")),
            )
        )
        if not load_last and len(list_ckpt) == 1:
            print(f"Loading best model: {list_ckpt[0]}.")
            checkpoint_path = os.path.join(
                log_path, list_matching[0], "files", list_ckpt[0]
            )
        else:
            print(f"Loading last checkpoint.")
            checkpoint_path = os.path.join(
                log_path, list_matching[0], "files", "last.ckpt"
            )
        config_path = os.path.join(
            log_path, list_matching[0], "files", "learning_config.py"
        )

        if config is None:
            config = config_argparse(config_path)
            distant_model_type = None
        else:
            distant_config = config_argparse(config_path)
            distant_model_type = distant_config.model_type
        config["load_from"] = log_id

        if config.model_type == "interaction_biased":
            dataloaders = WaymoDataloaders(config)
        else:
            [data_train, data_val, data_test] = load_create_dataset(config)
            dataloaders = SceneDataLoaders(
                state_dim=config.state_dim,
                num_steps=config.num_steps,
                num_steps_future=config.num_steps_future,
                batch_size=config.batch_size,
                data_train=data_train,
                data_val=data_val,
                data_test=data_test,
                num_workers=config.num_workers,
            )

        try:
            map_location = "cpu"
            model = load_weights(
                get_predictor(config, dataloaders.unnormalize_trajectory),
                torch.load(checkpoint_path, map_location=map_location),
                strict=True,
            )
        except RuntimeError:
            raise RuntimeError(
                f"The source model is of type {distant_model_type}."
                + " It cannot be used to load the weights of the interaction biased model."
            )

        return model, dataloaders, config

    else:
        print("Trying to download logs from WandB...")
        api = wandb.Api()
        run = api.run(entity + "/" + project + "/" + log_id)
        if run is not None:
            checkpoint_path = os.path.join(
                log_path, "downloaded_run-" + log_id, "files"
            )
            os.makedirs(checkpoint_path)
            for file in run.files():
                if file.name.endswith("ckpt") or file.name.endswith("config.py"):
                    file.download(checkpoint_path)
            return load_from_wandb_id(
                log_id, log_path, entity, project, config, load_last
            )
        else:
            raise RuntimeError(
                f"Error while loading checkpoint: Found {len(list_matching)} occurences of the given id {log_id} in the logs at {log_path}."
            )


def load_from_config(cfg: Config):
    """
    This function loads the predictor model and the data depending on which one is selected in the config.
    If a "load_from" field is not empty, then tries to load the pre-trained model from the checkpoint.
    The matching config file is loaded

    Args:
        cfg : Configuration that defines the model to be loaded

    Returns:
        loaded model and a new version of the config that is compatible with the checkpoint model that it could be loaded from
    """

    log_path = os.path.join(cfg.log_path, "wandb")
    ignored_keys = [
        "project",
        "dataset_parameters",
        "load_from",
        "force_config",
        "load_last",
    ]

    if "load_from" in cfg.keys() and cfg.load_from != "" and cfg.load_from:
        if "load_last" in cfg.keys():
            load_last = cfg["load_last"]
        else:
            load_last = False
        if cfg.force_config:
            warnings.warn(
                f"Using local configuration but loading from run {cfg.load_from}. Will fail if local configuration is not compatible."
            )
            predictor, dataloaders, config = load_from_wandb_id(
                log_id=cfg.load_from,
                log_path=log_path,
                entity=cfg.entity,
                project=cfg.project,
                config=cfg,
                load_last=load_last,
            )
        else:
            predictor, dataloaders, config = load_from_wandb_id(
                log_id=cfg.load_from,
                log_path=log_path,
                entity=cfg.entity,
                project=cfg.project,
                load_last=load_last,
            )
            difference = False
            warning_message = ""
            for key, item in cfg.items():
                try:
                    if config[key] != item:
                        if not difference:
                            warning_message += "When loading the model, the configuration was changed to match the configuration of the pre-trained model to be loaded.\n"
                            difference = True
                        if key not in ignored_keys:
                            warning_message += f"    The value of '{key}' is now '{config[key]}' instead of '{item}'."
                except KeyError:
                    if not difference:
                        warning_message += "When loading the model, the configuration was changed to match the configuration of the pre-trained model to be loaded."
                        difference = True
                    warning_message += f"    The parameter '{key}' with value '{item}' does not exist for the model you are loading from, it is added."
                    config[key] = item
            if warning_message != "":
                warnings.warn(warning_message)
        return predictor, dataloaders, config

    else:
        if cfg.model_type == "interaction_biased":
            dataloaders = WaymoDataloaders(cfg)
        else:
            [data_train, data_val, data_test] = load_create_dataset(cfg)
            dataloaders = SceneDataLoaders(
                state_dim=cfg.state_dim,
                num_steps=cfg.num_steps,
                num_steps_future=cfg.num_steps_future,
                batch_size=cfg.batch_size,
                data_train=data_train,
                data_val=data_val,
                data_test=data_test,
                num_workers=cfg.num_workers,
            )

        predictor = get_predictor(cfg, dataloaders.unnormalize_trajectory)
        return predictor, dataloaders, cfg
