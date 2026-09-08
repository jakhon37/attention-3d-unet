"""Losses: paper BCE, plus Dice/focal/Tversky trials."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if self.dice_weight == 0:
            return bce
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        inter = (probs * target).sum(dims)
        den = probs.sum(dims) + target.sum(dims)
        dice = 1 - (2 * inter + 1) / (den + 1)
        return self.bce_weight * bce + self.dice_weight * dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        return ((1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    """Emphasize false negatives (missed tumor), α=0.3 β=0.7 by default."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        t = target.float()
        dims = tuple(range(1, p.ndim))
        tp = (p * t).sum(dims)
        fp = (p * (1 - t)).sum(dims)
        fn = ((1 - p) * t).sum(dims)
        tv = (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)
        return (1 - tv).mean()


def build_loss(name: str) -> nn.Module:
    name = (name or "bce").lower()
    if name == "bce":
        return BCEDiceLoss(1.0, 0.0)
    if name in ("bce_dice", "dice_bce"):
        return BCEDiceLoss(0.5, 0.5)
    if name == "focal":
        return FocalLoss()
    if name == "tversky":
        return TverskyLoss()
    raise ValueError(f"unknown loss: {name}")
