"""
LPBO-DRNet — Main Model: LPBOBoundaryConditionedNet
SRP — Combines Backbone + Head together, containing no training/evaluation logic.
DIP — Implements IClassifier interface (from domain layer), ready to swap
      with any other backbone when necessary.
Extracted from Cell 7 of LPBO/1.ipynb (checkpoint achieving 98.15% accuracy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .tokenizer import LesionPyramidTokenizer, BoundaryConditionedCascadedHead


# ─────────────────────────────────────────────────────────────────────────────
# Backbone Adapter (EfficientNetV2-S)
# ─────────────────────────────────────────────────────────────────────────────
class LPBOBackboneAdapter(nn.Module):
    """
    Wrapper adapter for EfficientNetV2-S backbone + LesionPyramidTokenizer.

    SRP — Responsible solely for extracting backbone features and projecting them into proj_dim.
    OCP — Easily extensible to support alternative backbones (timm, ConvNeXt, etc.)
          without modifying BoundaryConditionedCascadedHead.

    Parameters
    ----------
    proj_dim : int
        Output projection dimension.
    pyramid_stage_ids : tuple[int]
        Stage indices of EfficientNetV2-S used for the Lesion Pyramid.
    pyramid_stage_channels : tuple[int]
        Output channel count corresponding to each stage.
    pyramid_top_ks : tuple[int]
        Number of selected top-k tokens at each stage.
    pyramid_score_temp : float
        Softmax temperature for attention scoring.
    """

    # EfficientNetV2-S defaults (RS-27 proven config)
    DEFAULT_STAGE_IDS: tuple = (3, 4, 6)
    DEFAULT_STAGE_CHANNELS: tuple = (64, 128, 256)
    DEFAULT_TOP_KS: tuple = (12, 12, 8)

    def __init__(
        self,
        proj_dim: int = 256,
        pyramid_stage_ids: tuple = DEFAULT_STAGE_IDS,
        pyramid_stage_channels: tuple = DEFAULT_STAGE_CHANNELS,
        pyramid_top_ks: tuple = DEFAULT_TOP_KS,
        pyramid_score_temp: float = 0.70,
    ) -> None:
        super().__init__()
        self.stage_ids = set(pyramid_stage_ids)

        # Load EfficientNetV2-S with ImageNet pretrained weights
        weights = tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        eff = tv_models.efficientnet_v2_s(weights=weights)
        self.features = eff.features
        self.global_channels: int = eff.classifier[1].in_features  # 1280

        # 1×1 projection + BatchNorm + GELU
        self.proj = nn.Sequential(
            nn.Conv2d(self.global_channels, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

        # Lesion Pyramid Tokenizer
        self.lesion_pyramid = LesionPyramidTokenizer(
            stage_channels=pyramid_stage_channels,
            proj_dim=proj_dim,
            top_ks=pyramid_top_ks,
            score_temp=pyramid_score_temp,
        )

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (combined_tokens, lesion_tokens, lesion_scores)
          combined_tokens = cat([global_tokens, lesion_tokens])
        """
        maps = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self.stage_ids:
                maps.append(x)
        last = x  # output of the final stage block

        # Global tokens from the last feature map
        global_tokens = self.proj(last).flatten(2).transpose(1, 2)  # (B, HW, D)

        # Lesion tokens from multi-stage feature maps
        lesion_tokens, lesion_scores = self.lesion_pyramid(maps)     # (B, K, D)

        combined = torch.cat([global_tokens, lesion_tokens], dim=1)  # (B, HW+K, D)
        return combined, lesion_tokens, lesion_scores

    def backbone_parameters(self):
        return self.features.parameters()

    def head_parameters(self):
        return list(self.proj.parameters()) + list(self.lesion_pyramid.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Full LPBO-DRNet Model
# ─────────────────────────────────────────────────────────────────────────────
class LPBOBoundaryConditionedNet(nn.Module):
    """
    Complete DR classification model: EfficientNetV2-S + Lesion Pyramid + Boundary Head.

    This architecture achieved 98.15% test accuracy on APTOS 2019 + BLIP-GAN (RS_27).

    Parameters
    ----------
    num_classes : int        Number of DR grades (5).
    proj_dim : int           Embedding projection dimension (256).
    num_super_tokens : int   Number of super-tokens after compressor (16).
    num_heads : int          Number of multi-head attention heads (8).
    sa_layers : int          Number of self-attention layers (2).
    dropout : float          Dropout probability (0.1).
    drop_path_rate : float   DropPath rate (0.1).
    """

    def __init__(
        self,
        num_classes: int = 5,
        proj_dim: int = 256,
        num_super_tokens: int = 16,
        num_heads: int = 8,
        sa_layers: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = LPBOBackboneAdapter(proj_dim=proj_dim)
        self.head = BoundaryConditionedCascadedHead(
            proj_dim=proj_dim,
            num_classes=num_classes,
            num_super_tokens=num_super_tokens,
            num_heads=num_heads,
            sa_layers=sa_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

    # ── Parameter groups (to configure distinct learning rates for backbone & head) ─
    def backbone_parameters(self):
        return self.backbone.backbone_parameters()

    def head_parameters(self):
        return self.backbone.head_parameters() + list(self.head.parameters())

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor, return_boundary: bool = False
    ) -> Any:
        tokens, lesion_tokens, scores = self.backbone.forward_features(x)
        out = self.head(tokens, lesion_tokens, return_boundary=return_boundary)
        if return_boundary and isinstance(out, dict):
            out["lesion_scores"] = scores
        return out

    # ── Inference helpers ────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Input image (B, 3, H, W) → normalized probabilities (B, num_classes)."""
        tokens, lesion_tokens, _ = self.backbone.forward_features(x)
        return self.head.predict_proba(tokens, lesion_tokens)

    @torch.no_grad()
    def predict_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input image (B, 3, H, W) → (probabilities, boundary_logits) for BAO post-processing."""
        tokens, lesion_tokens, _ = self.backbone.forward_features(x)
        return self.head.predict_outputs(tokens, lesion_tokens)

    # ── Checkpoint helpers ───────────────────────────────────────────────────
    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None) -> None:
        """Save model state + metadata to a .pt checkpoint file."""
        payload: Dict = {"model_state": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_from_checkpoint(
        cls, path: Path, device: Optional[torch.device] = None, **kwargs
    ) -> "LPBOBoundaryConditionedNet":
        """Initialize model and load state from a checkpoint file."""
        model = cls(**kwargs)
        ckpt = torch.load(path, map_location=device or "cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        if device:
            model.to(device)
        return model
