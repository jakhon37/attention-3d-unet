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
