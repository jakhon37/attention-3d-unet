"""Binary cross-entropy as in §4.3, plus a Dice term that helps sparse tumors."""

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
