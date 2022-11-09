"""Profiling script based on the tutorial at the link below.

https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

Usage notes:
- Because this script imports `load_from_config`, you have to run it from the
"scripts" directory (risk_biased/scripts).
- This loads the model configuration and data from "waymo_config.py", so
to try different model hyperparameters, update them in that file.
"""
import pathlib
import sys
import time

import fire
import torch

from risk_biased.utils.load_model import load_from_config
from risk_biased.utils.config_argparse import config_argparse

# TODO: Avoid these hardcoded paths (it's the convention in the existing scripts).
DIR = pathlib.Path(__file__).parent.resolve()
CONFIG_FILEPATH = str(
    (DIR / ".." / ".." / "risk_biased" / "config" / "learning_config.py").resolve()
)
WAYMO_CONFIG_FILEPATH = str(
    (DIR / ".." / ".." / "risk_biased" / "config" / "waymo_config.py").resolve()
)


def run_training(module, dataloader, num_steps, prof):
    for i, batch in enumerate(dataloader):
        # batch = tuple([item.cuda() for item in batch])
        if i == num_steps:
            break
        module.training_step(batch, i)
        prof.step()


def main(num_active_batches: int = 20, perf_dir: str = "./log/001"):
    """
    Args:
        num_active_batches: Number of batches to run in computing perf information.
            This is separate from the setup / warmup batches, hence the name.
        perf_dir: Directory in which to store the output profile file.
            See the tutorial linked above for how to visualize this output.
    """
    # Overwrite sys.argv so it doesn't mess up the parser.
    sys.argv = sys.argv[:1]
    cfg = config_argparse([CONFIG_FILEPATH, WAYMO_CONFIG_FILEPATH])
    predictor, dataloaders, cfg = load_from_config(cfg)
    predictor.cuda()

    # Determine number of batches for each step.
    wait = 2
    warmup = 3
    active = num_active_batches
    num_batches = wait + warmup + active

    # Profile training steps.
    print("Running profile...")
    start = time.time()
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=0
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(perf_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        run_training(predictor, dataloaders.train_dataloader(), num_batches, prof)
    end = time.time()
    avg = (end - start) / num_batches
    print(f"Average training step time: {avg:0.4f}")


if __name__ == "__main__":
    fire.Fire(main)
