"""
ACFD-GAN Generator & Attention Modules (ACFF, AMM)
SRP — Retinal lesion synthesis module with Background-Preserving Inpainting.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.infrastructure.deep_learning.models.blip_gan.lrdb import LRDB


class ACFF(nn.Module):
    """
    Adaptive Cross-scale Feature Fusion (skip connection fusion).
    Combines Channel Attention and Spatial Attention.
    """

    def __init__(self, ch: int, r: int = 8) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.ca = nn.Sequential(
            nn.Conv2d(ch * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        if dec_feat.shape[-2:] != enc_feat.shape[-2:]:
            dec_feat = F.interpolate(
                dec_feat, size=enc_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        fr = enc_feat + dec_feat
        avg = F.adaptive_avg_pool2d(fr, 1)
        mx = F.adaptive_max_pool2d(fr, 1)
        alpha = self.ca(torch.cat([avg, mx], dim=1))
        ca_feat = fr * alpha
        sa_in = torch.cat(
            [ca_feat.mean(1, keepdim=True), ca_feat.max(1, keepdim=True)[0]], dim=1
        )
        beta = self.sa(sa_in)
        weighted = ca_feat * (1.0 + beta)
        return self.out(torch.cat([dec_feat, weighted], dim=1))


class AMM(nn.Module):
    """
    Adaptive Modulation Module (AdaIN conditioned on mask + latent).
    """

    def __init__(self, ch: int, latent_dim: int, mask_ch: int = 3) -> None:
        super().__init__()
        hidden = max(ch, latent_dim)
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.mask_enc = nn.Sequential(
            nn.Conv2d(mask_ch, max(ch // 4, 8), 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 4, 8), ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mask_proj = nn.Linear(ch, latent_dim)
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, ch * 2),
        )

    def forward(self, feat: torch.Tensor, mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m = F.interpolate(mask, size=feat.shape[2:], mode="bilinear", align_corners=False)
        m = self.mask_enc(m).flatten(1)
        mask_latent = self.mask_proj(m)
        fused = torch.cat([z, mask_latent], dim=1)
        gamma, beta = self.mapping(fused).chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return (1.0 + gamma) * self.norm(feat) + beta


class ACFDGenerator(nn.Module):
    """
    ACFD-GAN Generator
    Input: (mask, z_concat) where z_concat = cat(wae_z, random_noise)
    Architecture: U-Net with LRDB encoder, ACFF skip, AMM+AdaIN decoder
    """

    def __init__(
        self,
        mask_ch: int = 3,
        img_ch: int = 3,
        latent_dim: int = 256,
        base_ch: int = 64,
        growth: int = 32,
        img_size: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._s4 = img_size // 16
        c = base_ch

        # Encoder input
        self.enc_in = nn.Sequential(
            nn.Conv2d(mask_ch, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                LRDB(in_ch, growth),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.enc1 = enc_block(c, c * 2)  # 256 -> 128
        self.enc2 = enc_block(c * 2, c * 4)  # 128 -> 64
        self.enc3 = enc_block(c * 4, c * 8)  # 64 -> 32
        self.enc4 = enc_block(c * 8, c * 8)  # 32 -> 16

        # Bottleneck latent injection
        flat_z = (c * 8) * self._s4 * self._s4
        self.z_proj = nn.Linear(latent_dim, flat_z)
        self.z_norm = nn.LayerNorm(flat_z)

        # Skip Connections & AMM
        self.acff4 = ACFF(c * 8)
        self.acff3 = ACFF(c * 4)
        self.acff2 = ACFF(c * 2)
        self.acff1 = ACFF(c)

        self.amm4 = AMM(c * 8, latent_dim, mask_ch)
        self.amm3 = AMM(c * 4, latent_dim, mask_ch)
        self.amm2 = AMM(c * 2, latent_dim, mask_ch)
        self.amm1 = AMM(c, latent_dim, mask_ch)

        def dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.dec4 = dec_block(c * 8, c * 4)
        self.dec3 = dec_block(c * 4, c * 2)
        self.dec2 = dec_block(c * 2, c)
        self.dec1 = dec_block(c, c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(c, img_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def _up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Encode
        e0 = self.enc_in(mask)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Latent injection
        zp = F.relu(self.z_norm(self.z_proj(z)))
        zp = zp.view(z.size(0), -1, self._s4, self._s4)
        h = e4 + zp

        # Decode with ACFF + AMM + AdaIN
        h = self._up(h)
        h = self.acff4(e3, h)
        h = self.amm4(h, mask, z)
        h = self.dec4(h)

        h = self._up(h)
        h = self.acff3(e2, h)
        h = self.amm3(h, mask, z)
        h = self.dec3(h)

        h = self._up(h)
        h = self.acff2(e1, h)
        h = self.amm2(h, mask, z)
        h = self.dec2(h)

        h = self._up(h)
        h = self.acff1(e0, h)
        h = self.amm1(h, mask, z)
        h = self.dec1(h)

        return self.out_conv(h)
