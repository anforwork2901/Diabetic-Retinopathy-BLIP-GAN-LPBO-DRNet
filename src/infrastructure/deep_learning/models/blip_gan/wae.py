"""
Wasserstein Autoencoder (WAE) for Latent Lesion Distribution Learning.
SRP — Models continuous latent distributions of retinal lesion features via MMD regularization.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class WAEEncoder(nn.Module):
    """WAE Encoder: 4x Conv(stride 2) + BN + LeakyReLU -> FC -> latent z."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 128, img_size: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat = 512 * (img_size // 16) * (img_size // 16)
        self.fc = nn.Linear(flat, latent_dim)
        self._flat = flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class WAEDecoder(nn.Module):
    """WAE Decoder: FC -> reshape -> 4x ConvTranspose -> Tanh."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 128, img_size: int = 256) -> None:
        super().__init__()
        self._s = img_size // 16
        flat = 512 * self._s * self._s
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flat),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 512, self._s, self._s)
        return self.deconv(h)


class WAE(nn.Module):
    """Full WAE: Autoencoder + MMD Regularizer."""

    def __init__(self, img_channels: int = 3, latent_dim: int = 128, img_size: int = 256) -> None:
        super().__init__()
        self.encoder = WAEEncoder(img_channels, latent_dim, img_size)
        self.decoder = WAEDecoder(img_channels, latent_dim, img_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def mmd_loss(
    z: torch.Tensor,
    reg_weight: float = 10.0,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    """Unbiased multi-scale RBF Maximum Mean Discrepancy between z and N(0,1)."""
    z = z.float()
    B = z.size(0)
    z_prior = torch.randn_like(z)

    def pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = (x**2).sum(1, keepdim=True)
        yy = (y**2).sum(1, keepdim=True)
        return (xx + yy.t() - 2 * (x @ y.t())).clamp(min=0.0)

    d_zz = pdist2(z, z)
    d_pp = pdist2(z_prior, z_prior)
    d_zp = pdist2(z, z_prior)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)

    mmd = torch.tensor(0.0, device=z.device)
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma**2)
        k_zz = torch.exp(-d_zz * gamma).masked_fill(eye, 0).sum() / max(1, B * (B - 1))
        k_pp = torch.exp(-d_pp * gamma).masked_fill(eye, 0).sum() / max(1, B * (B - 1))
        k_zp = torch.exp(-d_zp * gamma).mean()
        mmd = mmd + k_zz + k_pp - 2 * k_zp

    return reg_weight * mmd / len(sigmas)
