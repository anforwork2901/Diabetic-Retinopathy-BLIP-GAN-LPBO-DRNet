"""
LPBO-DRNet — Lesion Pyramid Tokenizer & Boundary-Conditioned Cascaded Head
SRP — Each class encapsulates a single architectural responsibility.
OCP — Readily extensible to additional stages or heads without modifying existing code.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .blocks import CrossAttentionCompressor, OrdinalHead, make_ffn, DropPath


# ─────────────────────────────────────────────────────────────────────────────
# Lesion Pyramid Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class LesionPyramidTokenizer(nn.Module):
    """
    Extracts multi-scale lesion-aware visual tokens across backbone stages.

    For each stage s:
        1. Projects channel dimension from stage_channels[s] -> proj_dim via 1x1 conv.
        2. Flattens spatial dimensions -> (B, H*W, proj_dim).
        3. Selects top-k patches based on learned lesion attention scores.

    Outputs are concatenated across all stages into a unified lesion token sequence.

    Parameters
    ----------
    stage_channels : tuple[int]
        Output channel counts across feature stages (e.g., (64, 128, 256)).
    proj_dim : int
        Unified token projection dimension.
    top_ks : tuple[int]
        Number of tokens selected per stage (e.g., (12, 12, 8)).
    score_temp : float
        Softmax temperature for attention scoring.
    """

    def __init__(
        self,
        stage_channels: Tuple[int, ...],
        proj_dim: int,
        top_ks: Tuple[int, ...],
        score_temp: float = 0.70,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(top_ks)
        self.top_ks = top_ks
        self.score_temp = score_temp

        # Projection and scoring layers per stage
        self.proj_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, proj_dim, 1, bias=False),
                    nn.BatchNorm2d(proj_dim),
                    nn.GELU(),
                )
                for c in stage_channels
            ]
        )
        self.score_layers = nn.ModuleList(
            [nn.Conv2d(proj_dim, 1, 1, bias=True) for _ in stage_channels]
        )

    def forward(
        self, feature_maps: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature_maps : List[Tensor]
            List of feature map tensors from backbone stages, each (B, C_s, H_s, W_s).

        Returns
        -------
        lesion_tokens : Tensor (B, total_topk, proj_dim)
        lesion_scores : Tensor (B, total_topk)
        """
        all_tokens: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

        for fmap, proj, scorer, topk in zip(
            feature_maps, self.proj_layers, self.score_layers, self.top_ks
        ):
            feat = proj(fmap)                                   # (B, D, H, W)
            B, D, H, W = feat.shape
            score = scorer(feat).view(B, H * W)                 # (B, HW)
            score = F.softmax(score / self.score_temp, dim=-1)  # (B, HW)

            # Top-k selection by score
            actual_k = min(topk, H * W)
            topk_scores, topk_idx = score.topk(actual_k, dim=-1)  # (B, k)

            tokens = feat.view(B, D, H * W).permute(0, 2, 1)   # (B, HW, D)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
            selected = tokens.gather(1, idx_exp)                 # (B, k, D)

            all_tokens.append(selected)
            all_scores.append(topk_scores)

        lesion_tokens = torch.cat(all_tokens, dim=1)            # (B, total_k, D)
        lesion_scores = torch.cat(all_scores, dim=1)            # (B, total_k)
        return lesion_tokens, lesion_scores


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-Conditioned Cascaded Head
# ─────────────────────────────────────────────────────────────────────────────
class BoundaryConditionedCascadedHead(nn.Module):
    """
    Cascaded classification head combining:
      • CrossAttentionCompressor — compresses global+lesion tokens into K super-tokens.
      • OrdinalHead              — outputs K-1 binary logits (ordinal regression).
      • Boundary Heads           — outputs K-1 scalar logits for contiguous grade boundaries.
      • CE Aux Head              — auxiliary cross-entropy head.
      • Prototype Head           — class prototype cosine similarity head.

    Parameters
    ----------
    proj_dim : int           Token embedding dimension.
    num_classes : int        Number of target clinical classes (5).
    num_super_tokens : int   Number of super-tokens K after compressor (16).
    num_heads : int          Number of multi-head attention heads (8).
    sa_layers : int          Number of subsequent self-attention layers after compressor.
    dropout : float          Dropout probability.
    drop_path_rate : float   DropPath stochastic depth rate.
    """

    def __init__(
        self,
        proj_dim: int,
        num_classes: int,
        num_super_tokens: int = 16,
        num_heads: int = 8,
        sa_layers: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        K = num_super_tokens

        # Cross-Attention Compressor
        self.compressor = CrossAttentionCompressor(
            proj_dim, K, num_heads, dropout, drop_path_rate
        )

        # Self-Attention layers after compressor
        self.sa_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    proj_dim, num_heads, proj_dim * 4, dropout, batch_first=True
                )
                for _ in range(sa_layers)
            ]
        )

        # Ordinal Head: each super-token -> 1 binary threshold logit
        self.ordinal_head = OrdinalHead(proj_dim, num_classes, dropout)

        # Boundary Heads: K-1 heads, one per contiguous grade pair
        self.boundary_heads = nn.ModuleList(
            [nn.Linear(proj_dim, 1) for _ in range(num_classes - 1)]
        )
        self.boundary_norm = nn.LayerNorm(proj_dim)

        # Cross-attention boundary conditioning
        self.boundary_gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )

        # CE Auxiliary Head
        self.ce_head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

        # Prototype Head
        self.class_prototypes = nn.Parameter(
            torch.empty(num_classes, proj_dim)
        )
        nn.init.trunc_normal_(self.class_prototypes, std=0.02)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, N, D) → compressed super-tokens (B, K, D) → shared (B, D)."""
        x = self.compressor(tokens)                              # (B, K, D)
        for sa in self.sa_layers:
            x = sa(x)
        return x.mean(dim=1)                                     # (B, D) — global avg

    def _condition(
        self, shared: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditions shared representation with lesion context -> conditioned per-threshold representation."""
        context = lesion_tokens.mean(dim=1)                      # (B, D)
        gate = self.boundary_gate(
            torch.cat([shared, context], dim=1)
        ).unsqueeze(-1)                                           # (B, D, 1)
        conditioned = self.boundary_norm(
            shared.unsqueeze(1) + gate.squeeze(-1).unsqueeze(1)
        )                                                         # (B, 1, D)
        # Expand across K-1 super-tokens to map with K-1 boundary heads
        conditioned = conditioned.expand(
            -1, self.num_classes - 1, -1
        )                                                         # (B, K-1, D)
        return conditioned, gate.squeeze(-1)

    def _classify_conditioned(
        self, tokens: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self._encode(tokens)                             # (B, D)
        conditioned, gates = self._condition(shared, lesion_tokens)

        # Ordinal logits: each threshold k takes conditioned feature conditioned[:, k]
        all_threshold_logits = torch.stack(
            [self.ordinal_head(conditioned[:, k])[:, k] for k in range(conditioned.shape[1])],
            dim=1,
        )                                                         # (B, K-1)

        # Boundary logits
        boundary_logits = torch.stack(
            [self.boundary_heads[k](conditioned[:, k]) for k in range(conditioned.shape[1])],
            dim=1,
        ).squeeze(-1)                                             # (B, K-1)

        # CE auxiliary
        ce_logits = self.ce_head(shared)                         # (B, num_classes)

        # Prototype similarity
        PROTOTYPE_TEMP = 0.20
        shared_norm = F.normalize(shared, dim=1)
        proto_norm = F.normalize(self.class_prototypes, dim=1)
        prototype_logits = shared_norm @ proto_norm.t() / PROTOTYPE_TEMP  # (B, num_classes)

        return all_threshold_logits, boundary_logits, gates, ce_logits, prototype_logits

    def forward(
        self,
        tokens: torch.Tensor,
        lesion_tokens: torch.Tensor,
        return_boundary: bool = False,
    ) -> Any:
        ordinal_logits, boundary_logits, gates, ce_logits, prototype_logits = (
            self._classify_conditioned(tokens, lesion_tokens)
        )
        if not return_boundary:
            return ordinal_logits
        return {
            "ordinal_logits": ordinal_logits,
            "boundary_logits": boundary_logits,
            "boundary_gates": gates,
            "ce_logits": ce_logits,
            "prototype_logits": prototype_logits,
        }

    def predict_proba(
        self, tokens: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> torch.Tensor:
        """(B, N, D) → class probabilities (B, num_classes)."""
        logits, _, _, _, _ = self._classify_conditioned(tokens, lesion_tokens)
        return self.ordinal_head.logits_to_proba(logits)

    def predict_outputs(
        self, tokens: torch.Tensor, lesion_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (proba, boundary_logits) for test-time BAO post-processing."""
        logits, boundary, _, _, _ = self._classify_conditioned(tokens, lesion_tokens)
        return self.ordinal_head.logits_to_proba(logits), boundary
