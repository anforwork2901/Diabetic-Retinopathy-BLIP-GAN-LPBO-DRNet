"""
Domain Interfaces (Abstract Protocols)
ISP + DIP — Cohesive, specialized interfaces.
            Use Cases depend solely on domain abstractions, decoupled from concrete frameworks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ISP — Segregated interfaces exposing only what clients require
# ─────────────────────────────────────────────────────────────────────────────


class IClassifier(ABC):
    """
    Interface for all DR classification models.
    OCP — New backbones (DenseNet, LPBO-DRNet, Swin) implement this interface
          without requiring modifications to Use Cases.
    """

    @abstractmethod
    def predict_proba(self, images: Any) -> np.ndarray:
        """
        Predicts class probabilities for a batch of images.

        Parameters
        ----------
        images : Any
            Batch image tensor (B, C, H, W).

        Returns
        -------
        np.ndarray
            Probability matrix (B, num_classes).
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Saves model weights to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Loads model weights from disk."""
        ...


class IDatasetLoader(ABC):
    """
    Interface for dataset loaders.
    LSP — APTOSLoader and MessidorLoader are interchangeable across pipelines.
    """

    @abstractmethod
    def load_manifest(self) -> "pd.DataFrame":  # noqa: F821
        """
        Loads the entire dataset metadata manifest as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with required columns: ['image_path', 'label', 'source', 'id_code'].
        """
        ...

    @abstractmethod
    def get_split(
        self, split: str
    ) -> "pd.DataFrame":  # noqa: F821
        """
        Retrieves a specific subset split.

        Parameters
        ----------
        split : str
            'train', 'val', or 'test'.
        """
        ...


class IGenerator(ABC):
    """
    Interface for generative models (BLIP-GAN).
    ISP — Completely segregated from IClassifier.
    """

    @abstractmethod
    def generate(self, condition: Any, background: Any) -> Any:
        """
        Synthesizes novel lesion images conditioned on pseudo-masks and real backgrounds.

        Parameters
        ----------
        condition : Any
            4-channel pseudo-mask tensor.
        background : Any
            Real fundus background image tensor.

        Returns
        -------
        Any
            Synthetic fundus image tensor.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Saves generator weights to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Loads generator weights from disk."""
        ...


class IMetricsCalculator(ABC):
    """
    Interface for metric computation.
    SRP + ISP — Decouples metric evaluation from model inference.
    """

    @abstractmethod
    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes comprehensive evaluation metrics.

        Returns
        -------
        Dict[str, float]
        """
        ...


# Alias for backward compatibility
ClassifierInterface = IClassifier


