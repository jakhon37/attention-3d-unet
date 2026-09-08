"""Plain 3D U-Net baseline (Çiçek et al.), used as a comparison in Table 1."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenet3d import norm3d


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            norm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            norm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c_in = in_channels
        for c in chs:
            self.enc.append(DoubleConv(c_in, c))
            self.down.append(nn.Conv3d(c, c, 2, stride=2))
            c_in = c
        self.bot = DoubleConv(chs[-1], chs[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c_in = chs[-1] * 2
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose3d(c_in, c, 2, stride=2))
            self.dec.append(DoubleConv(c * 2, c))
            c_in = c
        self.head = nn.Conv3d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bot(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = dec(torch.cat([skip, h], dim=1))
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="trilinear", align_corners=False)
        return self.head(h)
