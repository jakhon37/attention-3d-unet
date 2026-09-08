"""Training loop matching §4.3, plus extra experiment flags."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import BraTS2020, RandomFlipIntensity, SyntheticBraTS
from .factory import build_model
from .losses import build_loss
from .metrics import evaluate_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = cfg_path.parent / base_path
        merged = load_config(str(base_path))
        merged.update(cfg)
        return merged
    return cfg


def build_loader(cfg: Dict[str, Any], train: bool = True) -> DataLoader:
    size = tuple(cfg.get("size", [16, 16, 16]))
    if cfg.get("synthetic", True):
        ds = SyntheticBraTS(n=int(cfg.get("synthetic_n", 8)), size=size, seed=int(cfg.get("seed", 0)))
    else:
        ds = BraTS2020(cfg["brats_root"], size=size)
    if train and cfg.get("augment"):
        ds = RandomFlipIntensity(ds)
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 0)),
    )


def _as_main_aux(
    out: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if isinstance(out, tuple):
        return out[0], list(out[1])
    return out, []


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
    loss_fn = build_loss(str(cfg.get("loss", "bce")))
    aux_w = float(cfg.get("aux_weight", 0.4))
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
            main, aux = _as_main_aux(model(image))
            loss = loss_fn(main, mask)
            for a in aux:
                loss = loss + aux_w * loss_fn(a, mask)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        epoch_loss = running / max(n, 1)
        sched.step(epoch_loss)
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            logits, _ = _as_main_aux(model(batch["image"].to(device)))
            metrics = evaluate_batch(logits, batch["mask"].to(device))
        line = (
            f"epoch {epoch:03d} model={cfg.get('model', 'paper')} "
            f"loss={epoch_loss:.4f} dice={metrics['dice']:.4f}"
        )
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
