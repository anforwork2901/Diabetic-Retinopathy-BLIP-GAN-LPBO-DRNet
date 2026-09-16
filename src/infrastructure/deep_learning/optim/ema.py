"""
EMA — Exponential Moving Average
SRP — Smooths model weights over training steps to prevent overfitting.
Extracted and refactored from research experiment notebooks.
RS_27 verified: EMA_DECAY = 0.9995
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a smoothed shadow copy of model weights:
        shadow = decay * shadow + (1 - decay) * current

    The EMA shadow copy consistently yields higher generalization accuracy
    and QWK on validation/test sets than raw final training weights.

    Parameters
    ----------
    model : nn.Module
        Target model to track.
    decay : float
        Decay smoothing factor. RS_27 uses 0.9995.
    device : Optional[torch.device]
        Target device storing shadow weights.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = decay
        self.device = device
        # Create detached shadow copy without autograd graph tracking
        self.shadow = deepcopy(model)
        self.shadow.eval()
        if self.device is not None:
            self.shadow.to(self.device)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Updates shadow weights after each training iteration."""
        for s_param, m_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict:
        """Returns shadow model state dict for checkpoint persistence."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores shadow weights from checkpoint state dict."""
        self.shadow.load_state_dict(state_dict)

    def apply_to(self, model: nn.Module) -> dict:
        """Applies shadow weights to target model for evaluation/testing.

        Returns
        -------
        dict
            Backup of original model state_dict for restoration after eval.
        """
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow.state_dict())
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        """Restores original model weights from backup state dict."""
        model.load_state_dict(backup)


# Alias
ModelEMA = EMA
