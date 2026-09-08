"""3D MobileNetV2 inverted residuals used as the encoder (§3.2)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def norm3d(ch: int) -> nn.Module:
    groups = 8 if ch >= 8 else 1
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


def _make_divisible(value: float, divisor: int = 8) -> int:
    new = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new < 0.9 * value:
        new += divisor
    return int(new)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
        relu6: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        layers: List[nn.Module] = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm3d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvertedResidual3D(nn.Module):
    """Expand 1×1×1 → depthwise 3×3×3 → project 1×1×1, residual when possible."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int) -> None:
        super().__init__()
        hidden = _make_divisible(in_ch * expand)
        self.use_res = stride == 1 and in_ch == out_ch
        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel=1, relu6=True))
        else:
            hidden = in_ch
        layers.extend(
            [
                ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden, relu6=True),
                ConvBNAct(hidden, out_ch, kernel=1, act=False),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res:
            out = out + x
        return out


# t, c, n, s  — same layout as MobileNetV2, used to produce 5 skip scales.
_CFG: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 2),
)


class MobileNetV2Encoder3D(nn.Module):
    """Five-scale 3D encoder. Returns feature maps from coarse to... no, fine first.

    Returns a list ``[e0, e1, e2, e3, e4]`` where ``e0`` is closest to
    input resolution and ``e4`` is the bottleneck.
    """

    def __init__(self, in_channels: int = 4, width_mult: float = 1.0) -> None:
        super().__init__()
        stem_ch = _make_divisible(32 * width_mult)
        self.stem = ConvBNAct(in_channels, stem_ch, kernel=3, stride=1, relu6=True)
        stages = []
        in_ch = stem_ch
        skip_channels: List[int] = []
        for expand, channels, n, stride in _CFG:
            out_ch = _make_divisible(channels * width_mult)
            blocks = [InvertedResidual3D(in_ch, out_ch, stride=stride, expand=expand)]
            for _ in range(n - 1):
                blocks.append(InvertedResidual3D(out_ch, out_ch, stride=1, expand=expand))
            stages.append(nn.Sequential(*blocks))
            skip_channels.append(out_ch)
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.skip_channels = skip_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
