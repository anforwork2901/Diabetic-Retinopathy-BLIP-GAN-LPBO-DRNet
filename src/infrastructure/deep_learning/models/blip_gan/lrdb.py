"""
Lightweight Residual Dense Block (LRDB)
SRP — Local feature extraction block employing Depthwise Separable Convolutions.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-Separable Convolution (3x3 depthwise + 1x1 pointwise)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LRDB(nn.Module):
    """Paper-style LRDB: reduced-width additive dense path + local residual fusion."""

    def __init__(self, in_ch: int, growth: int = 32) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, growth, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d1 = DepthwiseSeparableConv(growth)
        self.d2 = DepthwiseSeparableConv(growth)
        self.d3 = DepthwiseSeparableConv(growth)
        self.fuse = nn.Conv2d(in_ch + 4 * growth, in_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.reduce(x)
        f1 = self.d1(f0)
        f2 = self.d2(f0 + f1)
        f3 = self.d3(f0 + f1 + f2)
        fused = self.fuse(torch.cat([x, f0, f1, f2, f3], dim=1))
        return x + fused
