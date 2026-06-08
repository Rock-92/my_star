from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Loss functions for single-channel star segmentation."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        positive_weight: float | None = None,
        mode: str = "bce_dice",
        false_positive_weight: float = 10.0,
        false_negative_weight: float = 10.0,
        target_positive_threshold: float = 0.5,
        target_negative_threshold: float = 0.05,
        hard_negative_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.positive_weight = positive_weight
        self.mode = str(mode)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)
        self.target_positive_threshold = float(target_positive_threshold)
        self.target_negative_threshold = float(target_negative_threshold)
        self.hard_negative_threshold = float(hard_negative_threshold)

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_mask = target_mask.float()
        if self.mode == "error_focused":
            return self._error_focused_loss(logits, target_mask)

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

    def _error_focused_loss(
        self,
        logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prob = torch.sigmoid(logits)
        positive = target_mask >= self.target_positive_threshold
        negative = target_mask <= self.target_negative_threshold
        hard_negative = negative & (prob.detach() >= self.hard_negative_threshold)

        positive_loss_map = F.binary_cross_entropy_with_logits(
            logits,
            torch.ones_like(logits),
            reduction="none",
        )
        negative_loss_map = F.binary_cross_entropy_with_logits(
            logits,
            torch.zeros_like(logits),
            reduction="none",
        )

        positive_loss = masked_mean(positive_loss_map, positive)
        negative_loss = masked_mean(negative_loss_map, hard_negative)
        bce = self.false_negative_weight * positive_loss + self.false_positive_weight * negative_loss

        dice_target = positive.float()
        dice = dice_loss(prob, dice_target)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, {
            "loss": total.detach(),
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
            "positive_loss": positive_loss.detach(),
            "negative_loss": negative_loss.detach(),
            "active_negative_fraction": hard_negative.float().mean().detach(),
        }


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.float()
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    intersection = torch.sum(prob * target, dim=dims)
    denominator = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denom = torch.sum(mask)
    return torch.sum(values * mask) / torch.clamp(denom, min=1.0)


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
