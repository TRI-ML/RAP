import warnings

import torch
from torch import Tensor


@torch.jit.script
def torch_linspace(start: Tensor, stop: Tensor, num: int) -> torch.Tensor:
    """
    Copy-pasted from https://github.com/pytorch/pytorch/issues/61292
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def load_weights(
    model: torch.nn.Module, checkpoint: dict, strict=True
) -> torch.nn.Module:
    """This function is used instead of the one provided by pytorch lightning
     because for unexplained reasons, the pytorch lightning load function did
     not behave as intended: loading several times from the same checkpoint
     resulted in different loaded weight values...

    Args:
        model: a model in which new weights should be set
        checkpoint: a loaded pytorch checkpoint (probably resulting from torch.load(filename))
        strict: Default to True, wether to fail if

    """
    if not strict:
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
        }
        diff1 = checkpoint["state_dict"].keys() - model_dict.keys()
        if diff1:
            warnings.warn(
                f"Found keys {diff1} in checkpoint without any match in the model, ignoring corresponding values."
            )
        diff2 = model_dict.keys() - checkpoint["state_dict"].keys()
        if diff2:
            warnings.warn(
                f"Missing keys {diff2} from the checkpoint, the corresponding weights will keep their initial values."
            )
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
    else:
        model_dict = checkpoint["state_dict"]

    model.load_state_dict(model_dict, strict=strict)
    return model
