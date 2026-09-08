"""Dice, precision, recall, Hausdorff (§4.2). CPU-friendly on small volumes."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _to_numpy(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 5:
        mask = mask[0, 0]
    elif mask.ndim == 4:
        mask = mask[0]
    return mask.detach().cpu().numpy().astype(bool)


def dice_score(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    p = (torch.sigmoid(pred) > thresh).float()
    t = target.float()
    dims = tuple(range(1, p.ndim))
    inter = (p * t).sum(dims)
    den = p.sum(dims) + t.sum(dims)
    d = (2 * inter + 1e-6) / (den + 1e-6)
    return float(d.mean().item())


def precision_recall(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> Tuple[float, float]:
    p = (torch.sigmoid(pred) > thresh).float()
    t = target.float()
    dims = tuple(range(1, p.ndim))
    tp = (p * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    fn = ((1 - p) * t).sum(dims)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return float(prec.mean().item()), float(rec.mean().item())


def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """Max of directed surface distances. Empty masks return 0."""
    p = _to_numpy((torch.sigmoid(pred) > thresh).float())
    t = _to_numpy(target.float() > 0.5)
    p_pts = np.argwhere(p)
    t_pts = np.argwhere(t)
    if len(p_pts) == 0 or len(t_pts) == 0:
        return 0.0
    # Pairwise Euclidean; fine for the tiny CPU test volumes.
    d = np.sqrt(((p_pts[:, None, :] - t_pts[None, :, :]) ** 2).sum(axis=2))
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def evaluate_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prec, rec = precision_recall(logits, target)
    return {
        "dice": dice_score(logits, target),
        "hausdorff": hausdorff_distance(logits, target),
        "precision": prec,
        "recall": rec,
    }
