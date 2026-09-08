"""Forward/backward on every trial variant."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn3dunet.factory import build_model
from attn3dunet.losses import build_loss
from attn3dunet.train import load_config

SIZE = (16, 16, 16)
NAMES = [
    "paper",
    "no_attention",
    "no_fullscale",
    "unet3d",
    "se_skip",
    "trans_bottleneck",
    "deep_supervision",
]


def _tiny(name: str) -> dict:
    return {
        "model": name,
        "in_channels": 4,
        "out_channels": 1,
        "width_mult": 0.25,
        "fusion_channels": 16,
        "deep_supervision": name == "deep_supervision",
    }


def test_variants_forward_backward():
    x = torch.randn(1, 4, *SIZE)
    t = torch.zeros(1, 1, *SIZE)
    t[:, :, 4:12, 4:12, 4:12] = 1
    loss_fn = build_loss("bce")
    for name in NAMES:
        model = build_model(_tiny(name))
        model.train()
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape == (1, 1, *SIZE), name
        loss = loss_fn(logits, t)
        if isinstance(out, tuple):
            for aux in out[1]:
                loss = loss + 0.4 * loss_fn(aux, t)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), name


def test_losses_run():
    logits = torch.randn(1, 1, 8, 8, 8)
    target = torch.zeros(1, 1, 8, 8, 8)
    target[:, :, 2:6, 2:6, 2:6] = 1
    for name in ("bce", "bce_dice", "focal", "tversky"):
        val = build_loss(name)(logits, target)
        assert torch.isfinite(val), name


def test_experiment_yamls_load():
    exp_dir = ROOT / "configs" / "experiments"
    for path in sorted(exp_dir.glob("*.yaml")):
        cfg = load_config(str(path))
        assert "model" in cfg, path.name
        assert cfg.get("epochs", 0) >= 1
