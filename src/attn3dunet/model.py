"""Attention 3D U-Net with U-Net 3+ full-scale skips (paper Figure 1, §3).

Flags select paper ablations and extra trials without forking the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import make_skip_filter
from .bottleneck import TransformerBottleneck
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

    def __init__(
        self,
        in_channels: List[int],
        fusion: int,
        gate_ch: int,
        attention: str = "gate",
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv3d(c, fusion, kernel_size=1, bias=False) for c in in_channels]
        )
        fused = fusion * len(in_channels)
        self.fuse = nn.Sequential(
            ConvBNReLU(fused, fusion),
            ConvBNReLU(fusion, fusion),
        )
        self.att = make_skip_filter(attention, fusion, gate_ch)

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


class UNetDecoderBlock(nn.Module):
    """Same-scale skip only (no full-scale connections)."""

    def __init__(self, skip_ch: int, up_ch: int, fusion: int, attention: str) -> None:
        super().__init__()
        self.proj_skip = nn.Conv3d(skip_ch, fusion, 1, bias=False)
        self.proj_up = nn.Conv3d(up_ch, fusion, 1, bias=False)
        self.fuse = nn.Sequential(ConvBNReLU(fusion * 2, fusion), ConvBNReLU(fusion, fusion))
        self.att = make_skip_filter(attention, fusion, fusion)

    def forward(self, skip: torch.Tensor, deeper: torch.Tensor) -> torch.Tensor:
        up = self.proj_up(_align(deeper, skip.shape[2:]))
        sk = self.proj_skip(skip)
        fused = self.fuse(torch.cat([sk, up], dim=1))
        return self.att(fused, up)


class Attention3DUNet(nn.Module):
    """Paper architecture plus selectable trials.

    Input ``(B, 4, D, H, W)``. Output logits ``(B, 1, D, H, W)``, or
    ``(logits, aux_list)`` when ``deep_supervision`` is on and training.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        width_mult: float = 1.0,
        fusion_channels: int = 64,
        skip: str = "fullscale",
        attention: str = "gate",
        bottleneck: str = "conv",
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.deep_supervision = deep_supervision
        self.encoder = MobileNetV2Encoder3D(in_channels=in_channels, width_mult=width_mult)
        enc_ch = self.encoder.skip_channels
        n = len(enc_ch)
        if bottleneck == "transformer":
            self.bottleneck = TransformerBottleneck(enc_ch[-1], dropout=dropout)
        else:
            self.bottleneck = nn.Sequential(
                ConvBNReLU(enc_ch[-1], enc_ch[-1]),
                nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
                ConvBNReLU(enc_ch[-1], enc_ch[-1]),
            )
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        blocks: List[nn.Module] = []
        for i in range(n - 2, -1, -1):
            n_deeper = n - 1 - i
            gate_ch = enc_ch[-1] if i == n - 2 else fusion_channels
            if skip == "unet":
                up_ch = enc_ch[-1] if i == n - 2 else fusion_channels
                blocks.append(
                    UNetDecoderBlock(enc_ch[i], up_ch, fusion_channels, attention)
                )
            else:
                part_ch = list(enc_ch) + [enc_ch[-1]] + [fusion_channels] * (n_deeper - 1)
                blocks.append(
                    FullScaleDecoderBlock(part_ch, fusion_channels, gate_ch, attention)
                )
        self.decoders = nn.ModuleList(blocks)
        self.head = nn.Conv3d(fusion_channels, out_channels, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                [nn.Conv3d(fusion_channels, out_channels, 1) for _ in range(len(blocks))]
            )
        else:
            self.aux_heads = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        in_size = x.shape[2:]
        enc = self.encoder(x)
        bot = self.drop(self.bottleneck(enc[-1]))
        decoded: List[torch.Tensor] = [bot]
        n = len(enc)
        for k, i in enumerate(range(n - 2, -1, -1)):
            if self.skip == "unet":
                d = self.decoders[k](enc[i], decoded[-1])
            else:
                d = self.decoders[k](list(enc) + decoded, enc[i].shape[2:], decoded[-1])
            decoded.append(d)
        out = decoded[-1]
        if out.shape[2:] != in_size:
            out = F.interpolate(out, size=in_size, mode="trilinear", align_corners=False)
        logits = self.head(out)
        if self.deep_supervision and self.training and self.aux_heads:
            aux = []
            for head, feat in zip(self.aux_heads, decoded[1:]):
                a = head(feat)
                if a.shape[2:] != in_size:
                    a = F.interpolate(a, size=in_size, mode="trilinear", align_corners=False)
                aux.append(a)
            return logits, aux
        return logits
