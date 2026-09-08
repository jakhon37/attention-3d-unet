"""Build models from a config name / flags."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .model import Attention3DUNet
from .unet3d import UNet3D


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    name = str(cfg.get("model", "paper")).lower()
    in_ch = int(cfg.get("in_channels", 4))
    out_ch = int(cfg.get("out_channels", 1))
    width = float(cfg.get("width_mult", 1.0))
    fusion = int(cfg.get("fusion_channels", 64))
    if name in ("unet3d", "unet", "baseline"):
        base = 8 if width <= 0.5 else 16
        return UNet3D(in_channels=in_ch, out_channels=out_ch, base=base)
    skip = str(cfg.get("skip", "fullscale"))
    attention = str(cfg.get("attention", "gate"))
    bottleneck = str(cfg.get("bottleneck", "conv"))
    if name == "no_attention":
        attention = "none"
    elif name == "no_fullscale":
        skip = "unet"
    elif name == "se_skip":
        attention = "se"
    elif name in ("trans_bottleneck", "transformer"):
        bottleneck = "transformer"
    elif name == "deep_supervision":
        cfg = dict(cfg)
        cfg["deep_supervision"] = True
    return Attention3DUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        width_mult=width,
        fusion_channels=fusion,
        skip=skip,
        attention=attention,
        bottleneck=bottleneck,
        dropout=float(cfg.get("dropout", 0.0)),
        deep_supervision=bool(cfg.get("deep_supervision", False)),
    )
