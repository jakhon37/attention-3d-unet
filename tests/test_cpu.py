"""CPU-only checks. Full BraTS training is GPU work."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attn3dunet.attention import AttentionGate3D
from attn3dunet.data import SyntheticBraTS
from attn3dunet.losses import BCEDiceLoss
from attn3dunet.metrics import dice_score, hausdorff_distance, precision_recall
from attn3dunet.mobilenet3d import InvertedResidual3D
from attn3dunet.model import Attention3DUNet
from attn3dunet.train import build_model, train


SIZE = (16, 16, 16)


def test_attention_gate_shapes():
    g = torch.randn(2, 8, 4, 4, 4)
    x = torch.randn(2, 16, 8, 8, 8)
    out = AttentionGate3D(16, 8, 8)(x, g)
    assert out.shape == x.shape
    assert (out.abs() <= x.abs() + 1e-5).all()


def test_inverted_residual_stride():
    x = torch.randn(1, 8, 8, 8, 8)
    y = InvertedResidual3D(8, 16, stride=2, expand=6)(x)
    assert y.shape == (1, 16, 4, 4, 4)
    z = InvertedResidual3D(8, 8, stride=1, expand=6)(x)
    assert z.shape == x.shape


def test_model_forward_cpu():
    model = Attention3DUNet(in_channels=4, out_channels=1, width_mult=0.25, fusion_channels=16)
    x = torch.randn(1, 4, *SIZE)
    y = model(x)
    assert y.shape == (1, 1, *SIZE)
    n = sum(p.numel() for p in model.parameters())
    assert n > 10_000


def test_model_backward_cpu():
    model = Attention3DUNet(in_channels=4, out_channels=1, width_mult=0.25, fusion_channels=16)
    x = torch.randn(1, 4, *SIZE)
    t = torch.zeros(1, 1, *SIZE)
    t[:, :, 4:12, 4:12, 4:12] = 1
    loss = BCEDiceLoss()(model(x), t)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0


def test_dice_perfect_and_empty():
    t = torch.zeros(1, 1, 8, 8, 8)
    t[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.where(t > 0, torch.tensor(10.0), torch.tensor(-10.0))
    assert dice_score(logits, t) > 0.99
    empty = torch.zeros_like(t)
    empty_logits = torch.full_like(t, -10.0)
    assert dice_score(empty_logits, empty) > 0.99


def test_precision_recall_and_hausdorff():
    t = torch.zeros(1, 1, 8, 8, 8)
    t[:, :, 2:5, 2:5, 2:5] = 1
    logits = torch.where(t > 0, torch.tensor(10.0), torch.tensor(-10.0))
    p, r = precision_recall(logits, t)
    assert p > 0.99 and r > 0.99
    assert hausdorff_distance(logits, t) == 0.0


def test_synthetic_dataset():
    ds = SyntheticBraTS(n=3, size=SIZE, seed=1)
    item = ds[0]
    assert item["image"].shape == (4, *SIZE)
    assert item["mask"].shape == (1, *SIZE)
    assert item["mask"].max() == 1


def test_one_train_epoch_cpu(tmp_path):
    cfg = {
        "seed": 0,
        "device": "cpu",
        "in_channels": 4,
        "out_channels": 1,
        "width_mult": 0.25,
        "fusion_channels": 16,
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.01,
        "lr_factor": 0.2,
        "lr_patience": 5,
        "size": list(SIZE),
        "synthetic": True,
        "synthetic_n": 2,
        "num_workers": 0,
        "out_dir": str(tmp_path),
    }
    out = train(cfg)
    assert (out / "last.pth").exists()
    ckpt = torch.load(out / "last.pth", map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        y = model(torch.randn(1, 4, *SIZE))
    assert y.shape[-3:] == SIZE
