"""Training loop matching §4.3 (Adam, lr 0.01, ReduceLROnPlateau, BCE)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import BraTS2020, SyntheticBraTS
from .losses import BCEDiceLoss
from .metrics import evaluate_batch
from .model import Attention3DUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_model(cfg: Dict[str, Any]) -> Attention3DUNet:
    return Attention3DUNet(
        in_channels=int(cfg.get("in_channels", 4)),
        out_channels=int(cfg.get("out_channels", 1)),
        width_mult=float(cfg.get("width_mult", 1.0)),
        fusion_channels=int(cfg.get("fusion_channels", 64)),
    )


def build_loader(cfg: Dict[str, Any], train: bool = True) -> DataLoader:
    size = tuple(cfg.get("size", [16, 16, 16]))
    if cfg.get("synthetic", True):
        ds = SyntheticBraTS(n=int(cfg.get("synthetic_n", 8)), size=size, seed=int(cfg.get("seed", 0)))
    else:
        ds = BraTS2020(cfg["brats_root"], size=size)
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 0)),
    )


def train(cfg: Dict[str, Any]) -> Path:
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cpu"))
    out_dir = Path(cfg.get("out_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.01)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.2)),
        patience=int(cfg.get("lr_patience", 5)),
    )
    loss_fn = BCEDiceLoss()
    loader = build_loader(cfg, train=True)

    best = float("inf")
    for epoch in range(1, int(cfg.get("epochs", 1)) + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(image)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        epoch_loss = running / max(n, 1)
        sched.step(epoch_loss)
        metrics = {}
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            metrics = evaluate_batch(model(batch["image"].to(device)), batch["mask"].to(device))
        line = f"epoch {epoch:03d} loss={epoch_loss:.4f} dice={metrics['dice']:.4f}"
        print(line)
        (out_dir / "train_log.txt").open("a", encoding="utf-8").write(line + "\n")
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": cfg,
            "loss": epoch_loss,
        }
        torch.save(ckpt, out_dir / "last.pth")
        if epoch_loss < best:
            best = epoch_loss
            torch.save(ckpt, out_dir / "best.pth")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Attention 3D U-Net")
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
