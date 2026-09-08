"""Optional transformer bottleneck (paper §6 future work)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBottleneck(nn.Module):
    """MHSA over bottleneck voxels. No-op if the map is a single voxel."""

    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        if d * h * w <= 1:
            return x
        tok = x.flatten(2).transpose(1, 2)
        nrm = self.norm(tok)
        att, _ = self.attn(nrm, nrm, nrm, need_weights=False)
        tok = tok + att
        tok = tok + self.ff(tok)
        return tok.transpose(1, 2).view(b, c, d, h, w)
