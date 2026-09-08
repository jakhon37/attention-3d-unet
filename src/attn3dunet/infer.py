"""Run a trained checkpoint on a tensor or a BraTS case."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .factory import build_model
from .train import load_config


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def predict(model, image: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    with torch.no_grad():
        out = model(image)
        logits = out[0] if isinstance(out, tuple) else out
        return (torch.sigmoid(logits) > thresh).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    model = load_checkpoint(args.ckpt, device)
    size = tuple(cfg.get("size", [16, 16, 16]))
    dummy = torch.zeros(1, 4, *size, device=device)
    mask = predict(model, dummy)
    print("pred shape", tuple(mask.shape), "foreground", int(mask.sum().item()))


if __name__ == "__main__":
    main()
