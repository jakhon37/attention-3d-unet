"""Attention 3D U-Net with U-Net 3+ full-scale skips (paper Figure 1, §3)."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionGate3D
from .mobilenet3d import MobileNetV2Encoder3D, norm3d


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _align(feat: torch.Tensor, size) -> torch.Tensor:
    if feat.shape[2:] == size:
        return feat
    if feat.shape[2] < size[0]:
        return F.interpolate(feat, size=size, mode="trilinear", align_corners=False)
    return F.adaptive_avg_pool3d(feat, output_size=size)


class FullScaleDecoderBlock(nn.Module):
    """U-Net 3+ fusion: every encoder scale + deeper decoder scales (§3.3)."""

    def __init__(self, in_channels: List[int], fusion: int, gate_ch: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv3d(c, fusion, kernel_size=1, bias=False) for c in in_channels]
        )
        fused = fusion * len(in_channels)
        self.fuse = nn.Sequential(
            ConvBNReLU(fused, fusion),
            ConvBNReLU(fusion, fusion),
        )
        self.att = AttentionGate3D(in_x=fusion, in_g=gate_ch, inter=max(fusion // 2, 4))

    def forward(
        self,
        parts: List[torch.Tensor],
        target_size,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        aligned = [
            proj(_align(p, target_size)) for proj, p in zip(self.projections, parts)
        ]
        fused = self.fuse(torch.cat(aligned, dim=1))
        return self.att(fused, gate)


class Attention3DUNet(nn.Module):
    """Paper architecture: MobileNetV2 encoder, full-scale skips, attention gates.

    Input ``(B, 4, D, H, W)`` BraTS modalities. Output logits ``(B, 1, D, H, W)``
    for whole-tumor binary segmentation (BCE, §4.3).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width_mult: float = 1.0,
        fusion_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = MobileNetV2Encoder3D(in_channels=in_channels, width_mult=width_mult)
        enc_ch = self.encoder.skip_channels
        n = len(enc_ch)
        self.bottleneck = nn.Sequential(
            ConvBNReLU(enc_ch[-1], enc_ch[-1]),
            ConvBNReLU(enc_ch[-1], enc_ch[-1]),
        )
        # Decode from scale n-2 down to 0. Each block sees all encoder maps
        # plus every deeper decoder map already produced.
        blocks = []
        for i in range(n - 2, -1, -1):
            n_deeper = n - 1 - i
            part_ch = list(enc_ch) + [enc_ch[-1]] + [fusion_channels] * (n_deeper - 1)
            gate_ch = enc_ch[-1] if i == n - 2 else fusion_channels
            blocks.append(
                FullScaleDecoderBlock(part_ch, fusion=fusion_channels, gate_ch=gate_ch)
            )
        self.decoders = nn.ModuleList(blocks)
        self.head = nn.Conv3d(fusion_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[2:]
        enc = self.encoder(x)
        bot = self.bottleneck(enc[-1])
        decoded: List[torch.Tensor] = [bot]
        n = len(enc)
        for k, i in enumerate(range(n - 2, -1, -1)):
            parts = list(enc) + decoded
            d = self.decoders[k](parts, enc[i].shape[2:], decoded[-1])
            decoded.append(d)
        out = decoded[-1]
        if out.shape[2:] != in_size:
            out = F.interpolate(out, size=in_size, mode="trilinear", align_corners=False)
        return self.head(out)
