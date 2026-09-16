"""
LPBO-DRNet — Building Blocks
SRP — Each class encapsulates a specific architectural component:
      DropPath, CrossAttentionCompressor, OrdinalHead.
OCP — Readily extensible with new blocks without altering existing implementations.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# DropPath (Stochastic Depth)
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drops structural paths during training.
    Employed within CrossAttentionCompressor to enhance regularization.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.floor(
            torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        )
        return x.div(keep_prob) * mask


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network helper
# ─────────────────────────────────────────────────────────────────────────────
def make_ffn(dim: int, expand: int = 4, dropout: float = 0.1) -> nn.Sequential:
    """Constructs a two-layer FFN with GELU activations."""
    return nn.Sequential(
        nn.Linear(dim, dim * expand),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expand, dim),
        nn.Dropout(dropout),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CrossAttentionCompressor (Perceiver IO-inspired)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionCompressor(nn.Module):
    """
    Compresses N spatial tokens from the backbone into K super-tokens
    using K learnable query vectors attending across N spatial tokens.

    K << N drastically reduces sequence length prior to self-attention.

    Input  : (B, N, C)
    Output : (B, K, C)
    """

    def __init__(
        self,
        embed_dim: int,
        num_super_tokens: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_super_tokens, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ffn = make_ffn(embed_dim, dropout=dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) → (B, K, C)"""
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)                      # (B, K, C)
        out, _ = self.cross(self.norm_q(q), self.norm_kv(x), self.norm_kv(x))
        q = q + self.dp(out)                                     # residual
        q = q + self.dp(self.ffn(self.norm_ff(q)))
        return q                                                  # (B, K, C)


# ─────────────────────────────────────────────────────────────────────────────
# OrdinalHead (Frank & Hall ordinal regression)
# ─────────────────────────────────────────────────────────────────────────────
class OrdinalHead(nn.Module):
    """
    Frank & Hall Ordinal Regression Head.

    Outputs K-1 binary logits representing P(y > k) for k = 0..K-2.
    ``logits_to_proba()`` translates binary logits into normalized K-class probabilities.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    num_classes : int
        Number of clinical classes K.
    dropout : float
        Dropout probability prior to linear projection.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature (B, C) → binary logits (B, K-1)."""
        return self.fc(x)

    def logits_to_proba(self, binary_logits: torch.Tensor) -> torch.Tensor:
        """
        Converts ordinal logits (B, K-1) into normalized class probabilities (B, K).

        Applies cumulative distribution difference:
            P(y=k) = P(y > k-1) - P(y > k)
        """
        p_gt = torch.sigmoid(binary_logits)                     # (B, K-1)
        ones = torch.ones(binary_logits.size(0), 1, device=binary_logits.device)
        zeros = torch.zeros(binary_logits.size(0), 1, device=binary_logits.device)
        cdf = torch.cat([ones, p_gt, zeros], dim=1)             # (B, K+1)
        probs = cdf[:, :-1] - cdf[:, 1:]                        # (B, K)
        probs = torch.clamp(probs, min=1e-7)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-7)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Feature (B, C) → class probabilities (B, K)."""
        return self.logits_to_proba(self.forward(x))
