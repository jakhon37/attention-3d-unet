"""3D attention gate (Oktay et al.), as used in §3.4 of the paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate3D(nn.Module):
    """Filter skip features ``x`` with a coarser gating signal ``g``.

    Paper Figure 4: 1×1×1 conv on both maps, add, ReLU, 1×1×1 conv,
    sigmoid, resample, multiply with ``x``.
    """

    def __init__(self, in_x: int, in_g: int, inter: int) -> None:
        super().__init__()
        self.W_x = nn.Sequential(
            nn.Conv3d(in_x, inter, kernel_size=1, bias=True),
            nn.GroupNorm(1, inter),
        )
        self.W_g = nn.Sequential(
            nn.Conv3d(in_g, inter, kernel_size=1, bias=True),
            nn.GroupNorm(1, inter),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter, 1, kernel_size=1, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="trilinear", align_corners=False)
        att = self.psi(self.relu(self.W_x(x) + self.W_g(g)))
        if att.shape[2:] != x.shape[2:]:
            att = F.interpolate(att, size=x.shape[2:], mode="trilinear", align_corners=False)
        return x * att


class IdentityGate(nn.Module):
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return x


class SE3D(nn.Module):
    """Channel squeeze-and-excitation on the skip (alternative to attention gates)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor = None) -> torch.Tensor:
        return x * self.fc(x)


def make_skip_filter(kind: str, in_x: int, in_g: int) -> nn.Module:
    if kind == "gate":
        return AttentionGate3D(in_x, in_g, inter=max(in_x // 2, 4))
    if kind == "se":
        return SE3D(in_x)
    if kind in ("none", "off", "identity"):
        return IdentityGate()
    raise ValueError(f"unknown attention kind: {kind}")
