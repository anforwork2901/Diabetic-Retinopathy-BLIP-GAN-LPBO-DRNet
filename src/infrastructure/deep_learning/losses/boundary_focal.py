"""
Loss Functions: Ordinal + Adjacent Boundary Focal Loss
SRP — Each loss module encapsulates a specific objective formulation.
OCP — New losses can be introduced without modifying existing loss implementations.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for ordinal regression (Frank & Hall formulation).

    For each threshold k in (0..K-2):
        y_k = 1 if true_label > k else 0.
    Loss = average BCE across all K-1 binary thresholds.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor. 0.0 disables smoothing.
    """

    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        binary_logits: torch.Tensor,   # (B, K-1)
        targets: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        K = binary_logits.shape[1] + 1
        ord_targets = torch.stack(
            [(targets > k).float() for k in range(K - 1)], dim=1
        )
        if self.smoothing > 0:
            ord_targets = ord_targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(binary_logits, ord_targets)


class OrdinalExpectedRankFocalLoss(nn.Module):
    """
    Focal-weighted Expected Rank Loss for ordinal logits.

    Focuses gradient updates on samples with large rank prediction discrepancies.

    Parameters
    ----------
    gamma : float
        Focal focusing parameter. Higher values emphasize harder samples.
    alpha : Optional[torch.Tensor]
        Class weights to compensate for severe class imbalance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        binary_logits: torch.Tensor,   # (B, K-1)
        targets: torch.Tensor,          # (B,)
        ordinal_head: nn.Module,        # Requires logits_to_proba() method
    ) -> torch.Tensor:
        K = binary_logits.shape[1] + 1
        probs = ordinal_head.logits_to_proba(binary_logits)           # (B, K)
        ranks = torch.arange(K, dtype=torch.float32, device=binary_logits.device)
        exp_r = (probs * ranks).sum(dim=1)                             # (B,)
        err = (exp_r - targets.float()).abs()
        focal = (1.0 - torch.exp(-err)) ** self.gamma
        if self.alpha is not None:
            at = self.alpha.to(binary_logits.device)[targets]
            focal = focal * at
        return (focal * err).mean()


class CombinedOrdinalLoss(nn.Module):
    """
    Combines OrdinalBCELoss and OrdinalExpectedRankFocalLoss.

    Parameters
    ----------
    gamma : float
        Focal gamma parameter for ExpectedRankFocalLoss.
    alpha : Optional[torch.Tensor]
        Class weighting tensor.
    ordinal_bce_weight : float
        Weight of the BCE component (remaining allocated to Expected Rank Focal).
    smoothing : float
        Label smoothing coefficient.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ordinal_bce_weight: float = 0.60,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_loss = OrdinalBCELoss(smoothing=smoothing)
        self.focal_rank = OrdinalExpectedRankFocalLoss(gamma, alpha)
        self.w_bce = ordinal_bce_weight

    def forward(
        self,
        binary_logits: torch.Tensor,
        targets: torch.Tensor,
        ordinal_head: nn.Module,
    ) -> torch.Tensor:
        l_bce = self.bce_loss(binary_logits, targets)
        l_focal = self.focal_rank(binary_logits, targets, ordinal_head)
        return self.w_bce * l_bce + (1.0 - self.w_bce) * l_focal


class AdjacentBoundaryFocalLoss(nn.Module):
    """
    Focal BCE applied selectively to samples belonging to contiguous grade pairs.

    Down-weights trivial non-adjacent decisions, forcing the model to concentrate
    on clinically ambiguous grade transitions (0-1, 1-2, 2-3, 3-4).

    Parameters
    ----------
    pair_weights : Optional[List[float]]
        Specialized weights for each adjacent pair (0-1, 1-2, 2-3, 3-4).
        RS_27 configuration uses [2.00, 2.50, 2.00, 1.50].
    gamma : float
        Focal focusing parameter. RS_27 uses 2.0.
    """

    def __init__(
        self,
        pair_weights: Optional[List[float]] = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.pair_weights = pair_weights
        self.gamma = float(gamma)

    def forward(
        self,
        boundary_logits: torch.Tensor,  # (B, K-1)
        targets: torch.Tensor,           # (B,)
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for k in range(boundary_logits.shape[1]):
            mask = (targets == k) | (targets == k + 1)
            if mask.any():
                pair_target = (targets[mask] == (k + 1)).float()
                pair_bce = F.binary_cross_entropy_with_logits(
                    boundary_logits[mask, k], pair_target, reduction="none"
                )
                pair_pt = torch.exp(-pair_bce)
                pair_loss = ((1.0 - pair_pt) ** self.gamma * pair_bce).mean()
                if self.pair_weights is not None:
                    pair_loss = pair_loss * float(self.pair_weights[k])
                losses.append(pair_loss)
        if not losses:
            return boundary_logits.sum() * 0.0
        return torch.stack(losses).mean()

