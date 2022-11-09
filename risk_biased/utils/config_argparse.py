import os
from typing import Optional, Union, List
import warnings

import argparse
from mmcv import Config


def config_argparse(config_path: Optional[Union[str, List[str]]] = None) -> Config:
    """Function that loads the config file as an MMCV Config object and overwrites its values with argparsed arguments.

    Args:
        config_path : path of the mmcv config file

    Returns:
        MMCV Config object with default values from the config_path and overwritten values from argparse
    """
    if config_path is None:
        working_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(
            working_dir, "..", "..", "config", "learning_config.py"
        )
    if isinstance(config_path, str):
        cfg = Config.fromfile(config_path)
    else:
        cfg = Config.fromfile(config_path[0])
        for path in config_path[1:]:
            c = Config.fromfile(path)
            cfg.update(c)

    parser = argparse.ArgumentParser()
    excluded_args = ["force_config", "load_last"]
    overwritable_types = (str, float, int, list)
    for key, value in cfg.items():
        if key not in excluded_args:
            if list in overwritable_types and isinstance(value, list):
                if len(value) > 0:
                    parser.add_argument(
                        "--" + key, default=value, nargs="+", type=type(value[0])
                    )
                else:
                    parser.add_argument("--" + key, default=value, nargs="+")
            elif isinstance(value, overwritable_types):
                parser.add_argument("--" + key, default=value, type=type(value))

    if "load_from" not in cfg.keys():
        parser.add_argument(
            "--load_from",
            default="",
            type=str,
            help="""Use this to load the model weights from a wandb checkpoint,
                    refer to the checkpoint with the wandb id (example:'1f1ho81a')""",
        )

    parser.add_argument(
        "--load_last",
        action="store_true",
        help="""Use this flag to force the use of the last checkpoint instead of the best one
        when loading a model.""",
    )

    parser.add_argument(
        "--force_config",
        action="store_true",
        help="""Use this flag to force the use of the local config file
        when loading a model from a checkpoint. Otherwise the checkpoint config file is used.
        In any case the parameters can be overwritten with an argparse argument.""",
    )
    if "force_config" not in cfg.keys():
        parser.set_defaults(force_config=False)
    else:
        parser.set_defaults(force_config=cfg.force_config)

    if "load_last" not in cfg.keys():
        parser.set_defaults(load_last=False)
    else:
        parser.set_defaults(force_config=cfg.force_config)

    args = parser.parse_args()

    # Print a warning in case the parameter 'dt' or 'time_scene' is changed becaus 'sample_times' might need to be updated accordingly.
    if (
        args.dt != cfg.dt or args.time_scene != cfg.time_scene
    ) and args.sample_times == cfg.sample_times:
        warnings.warn(
            f"""Parameter 'dt' has been changed from {args.dataset_parameters['dt']} to {args.dt} by
            a command line argument, it might be used to set the parameter 'sample_times' that
            cannot be updated accordingly. Consider setting 'dt' in {config_path} instead."""
        )
    # Config has a dataset_parameters field that copies the parameters related to dataset to compare them,
    # they must be updated too if some of the dataset parameters were changed by argparse
    for key, value in cfg.dataset_parameters.items():
        if isinstance(value, overwritable_types):
            cfg.dataset_parameters[key] = args.__getattribute__(key)
    cfg.update(args.__dict__)
    return cfg
