from typing import Optional

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal


def reconstruction_loss(
    pred: torch.Tensor, truth: torch.Tensor, mask_loss: Optional[torch.Tensor] = None
):
    """
    pred (Tensor): (..., time, [x,y,(a),(vx,vy)])
    truth (Tensor): (..., time, [x,y,(a),(vx,vy)])
    mask_loss (Tensor): (..., time) Defaults to None.
    """
    min_feat_shape = min(pred.shape[-1], truth.shape[-1])
    if min_feat_shape == 3:
        assert pred.shape[-1] == truth.shape[-1]
        return reconstruction_loss(
            pred[..., :2], truth[..., :2], mask_loss
        ) + reconstruction_loss(
            torch.stack([torch.cos(pred[..., 2]), torch.sin(pred[..., 2])], -1),
            torch.stack([torch.cos(truth[..., 2]), torch.sin(truth[..., 2])], -1),
            mask_loss,
        )
    elif min_feat_shape >= 5:
        assert pred.shape[-1] <= truth.shape[-1]
        v_norm = torch.sum(torch.square(truth[..., 3:5]), -1, keepdim=True)
        v_mask = v_norm > 1
        return (
            reconstruction_loss(pred[..., :2], truth[..., :2], mask_loss)
            + reconstruction_loss(
                torch.stack([torch.cos(pred[..., 2]), torch.sin(pred[..., 2])], -1)
                * v_mask,
                torch.stack([torch.cos(truth[..., 2]), torch.sin(truth[..., 2])], -1)
                * v_mask,
                mask_loss,
            )
            + reconstruction_loss(pred[..., 3:5], truth[..., 3:5], mask_loss)
        )
    elif min_feat_shape == 2:
        if mask_loss is None:
            return torch.mean(
                torch.sqrt(
                    torch.sum(
                        torch.square(pred[..., :2] - truth[..., :2]), -1
                    ).clamp_min(1e-6)
                )
            )
        else:
            assert mask_loss.any()
            mask_loss = mask_loss.float()
            return torch.sum(
                torch.sqrt(
                    torch.sum(
                        torch.square(pred[..., :2] - truth[..., :2]), -1
                    ).clamp_min(1e-6)
                )
                * mask_loss
            ) / torch.sum(mask_loss).clamp_min(1)


def map_penalized_reconstruction_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    map: torch.Tensor,
    mask_map: torch.Tensor,
    mask_loss: Optional[torch.Tensor] = None,
    map_importance: float = 0.1,
):
    """
    pred (Tensor): (batch_size, num_agents, time, [x,y,(a),(vx,vy)])
    truth (Tensor): (batch_size, num_agents, time, [x,y,(a),(vx,vy)])
    map (Tensor): (batch_size, num_objects, object_sequence_length, [x, y, ...])
    mask_map (Tensor): (...)
    mask_loss (Tensor): (..., time) Defaults to None.

    """
    #        b,  a,   o,  s,  f         b, a,  o,   t,  s,    f
    map_distance, _ = (
        (map[:, None, :, :, :2] - pred[:, :, None, -1, None, :2])
        .square()
        .sum(-1)
        .min(2)
    )
    map_distance = map_distance.sqrt().clamp(0.5, 3)
    if mask_map is not None:
        map_loss = (map_distance * mask_loss[..., -1:]).sum() / mask_loss[..., -1].sum()
    else:
        map_loss = map_distance.mean()

    rec_loss = reconstruction_loss(pred, truth, mask_loss)

    return rec_loss + map_importance * map_loss


def cce_loss_with_logits(pred_logits: torch.Tensor, truth: torch.Tensor):
    pred_log = pred_logits.log_softmax(-1)
    return -(pred_log * truth).sum(-1).mean()


def risk_loss_function(
    pred: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    factor: float = 100.0,
) -> torch.Tensor:
    """
    Loss function for the risk comparison. This is assymetric because it is preferred that the model over-estimates
    the risk rather than under-estimate it.
    Args:
        pred: (same_shape) The predicted risks
        truth: (same_shape) The reference risks to match
        mask: (same_shape) A mask with 1 where the loss should be computed and 0 elsewhere.
        approximate_mean_error: An approximation of the mean error obtained after training. The lower this value,
            the greater the intensity of the assymetry.
    Returns:
        Scalar loss value
    """
    error = pred - truth
    error = error * factor
    error = torch.where(error > 1, (error + 1e-6).log(), error.abs())
    error = (error * mask).sum() / mask.sum()
    return error
