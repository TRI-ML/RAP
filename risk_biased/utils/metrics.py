from typing import Optional

import torch


def FDE(
    pred: torch.Tensor, truth: torch.Tensor, mask_loss: Optional[torch.Tensor] = None
):
    """
    pred (Tensor): (..., time, xy)
    truth (Tensor): (..., time, xy)
    mask_loss (Tensor): (..., time) Defaults to None.
    """
    if mask_loss is None:
        return torch.mean(
            torch.sqrt(
                torch.sum(torch.square(pred[..., -1, :2] - truth[..., -1, :2]), -1)
            )
        )
    else:
        mask_loss = mask_loss.float()
        return torch.sum(
            torch.sqrt(
                torch.sum(torch.square(pred[..., -1, :2] - truth[..., -1, :2]), -1)
            )
            * mask_loss[..., -1]
        ) / torch.sum(mask_loss[..., -1]).clamp_min(1)


def ADE(
    pred: torch.Tensor, truth: torch.Tensor, mask_loss: Optional[torch.Tensor] = None
):
    """
    pred (Tensor): (..., time, xy)
    truth (Tensor): (..., time, xy)
    mask_loss (Tensor): (..., time) Defaults to None.
    """
    if mask_loss is None:
        return torch.mean(
            torch.sqrt(
                torch.sum(torch.square(pred[..., :, :2] - truth[..., :, :2]), -1)
            )
        )
    else:
        mask_loss = mask_loss.float()
        return torch.sum(
            torch.sqrt(
                torch.sum(torch.square(pred[..., :, :2] - truth[..., :, :2]), -1)
            )
            * mask_loss
        ) / torch.sum(mask_loss).clamp_min(1)


def minFDE(
    pred: torch.Tensor, truth: torch.Tensor, mask_loss: Optional[torch.Tensor] = None
):
    """
    pred (Tensor): (..., n_samples, time, xy)
    truth (Tensor): (..., time, xy)
    mask_loss (Tensor): (..., time) Defaults to None.
    """
    if mask_loss is None:
        min_distances, _ = torch.min(
            torch.sqrt(
                torch.sum(torch.square(pred[..., -1, :2] - truth[..., -1, :2]), -1)
            ),
            -1,
        )
        return torch.mean(min_distances)
    else:
        mask_loss = mask_loss[..., -1].float()
        final_distances = torch.sqrt(
            torch.sum(torch.square(pred[..., -1, :2] - truth[..., -1, :2]), -1)
        )
        max_final_distance = torch.max(final_distances * mask_loss)
        min_distances, _ = torch.min(
            final_distances + max_final_distance * (1 - mask_loss), -1
        )
        return torch.sum(min_distances * mask_loss.any(-1)) / torch.sum(
            mask_loss.any(-1)
        ).clamp_min(1)
