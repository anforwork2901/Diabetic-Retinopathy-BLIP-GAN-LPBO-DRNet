"""
Checkpoint Gateway: Safe Model State Serialization and Deserialization
SRP — Responsible exclusively for checkpoint I/O, independent of train/eval logic.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointGateway:
    """
    Manages safe model checkpoint persistence and retrieval.

    Features:
    - Supports multiple checkpoint variants: selected_primary, best_loss, etc.
    - Safe loading with automatic fallback to weights_only=False for legacy checkpoints.
    - Seamless integration with Exponential Moving Average (EMA) shadow weights.
    """

    def __init__(self, output_dir: Path = Path("checkpoints")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        filename: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
        ema: Optional[Any] = None,
        ema_state: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """
        Saves model state_dict + metadata into output_dir/filename or specific path.
        """
        payload: Dict[str, Any] = {"model_state": model.state_dict()}
        if metadata:
            payload.update(metadata)
        if ema is not None and hasattr(ema, "state_dict"):
            payload["ema_state"] = ema.state_dict()
        elif ema_state is not None:
            payload["ema_state"] = ema_state
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()

        target_path = Path(filename)
        if not target_path.is_absolute():
            # Only prepend output_dir if filename is a pure filename or doesn't start with output_dir
            if len(target_path.parts) == 1:
                target_path = self.output_dir / target_path
            elif not str(target_path).startswith(str(self.output_dir)):
                target_path = self.output_dir / target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target_path)
        return target_path

    def load(
        self, path: Path, map_location: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Loads checkpoint payload from file path.

        Automatically falls back to weights_only=False if PyTorch safe loading fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            warnings.warn(
                f"Loading checkpoint with weights_only=True failed. "
                f"Falling back for {path.name}.",
                stacklevel=2,
            )
            return torch.load(path, map_location=map_location, weights_only=False)

    def load_model_state(
        self,
        model: nn.Module,
        path: Path,
        map_location: Optional[Any] = None,
        use_ema: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads state_dict into model from checkpoint.

        Prioritizes EMA shadow weights if ``use_ema=True`` and 'ema_state' exists in checkpoint.

        Parameters
        ----------
        model : nn.Module    Target model instance.
        path : Path          Path to checkpoint file.
        map_location         Target device mapping.
        use_ema : bool       Whether to load EMA shadow weights if available.

        Returns
        -------
        Dict    Full checkpoint dictionary payload.
        """
        ckpt = self.load(path, map_location=map_location)
        if use_ema and "ema_state" in ckpt:
            model.load_state_dict(ckpt["ema_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        return ckpt

    def best_checkpoint_path(self, name: str = "selected_primary") -> Path:
        """Returns canonical checkpoint path by identifier name."""
        return self.output_dir / f"{name}.pt"
