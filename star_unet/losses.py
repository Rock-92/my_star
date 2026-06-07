from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """BCE + Dice loss for single-channel star segmentation."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        positive_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.positive_weight = positive_weight

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_mask = target_mask.float()
        pos_weight = None
        if self.positive_weight is not None and self.positive_weight > 0:
            pos_weight = torch.as_tensor(float(self.positive_weight), device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, target_mask, pos_weight=pos_weight)
        dice = dice_loss(torch.sigmoid(logits), target_mask)
        total = self.bce_weight * bce + self.dice_weight * dice

        return total, {
            "loss": total.detach(),
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
        }


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.float()
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    intersection = torch.sum(prob * target, dim=dims)
    denominator = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    target = target_mask >= 0.5

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}
