"""
ACFD-GAN Discriminator (5-layer PatchGAN with Spectral Normalization)
SRP — Evaluates realism of synthetic retinal lesion images against real fundus distributions.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ACFDDiscriminator(nn.Module):
    """
    ACFD-GAN Discriminator — 5-layer PatchGAN.
    3 stride-2 layers, 2 stride-1 layers.
    """

    def __init__(
        self,
        img_ch: int = 3,
        mask_ch: int = 3,
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        in_ch = img_ch + mask_ch

        def d_block(ic: int, oc: int, stride: int, first: bool = False) -> nn.Sequential:
            layers = [spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1, bias=False))]
            if not first:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        c = base_ch
        self.net = nn.Sequential(
            d_block(in_ch, c, 2, first=True),  # 256 -> 128
            d_block(c, c * 2, 2),              # 128 -> 64
            d_block(c * 2, c * 4, 2),          # 64 -> 32
            d_block(c * 4, c * 8, 1),          # 32 -> 32
            spectral_norm(nn.Conv2d(c * 8, 1, 4, 1, 1, bias=False)),  # patch output
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img, mask], dim=1)
        return self.net(x)
