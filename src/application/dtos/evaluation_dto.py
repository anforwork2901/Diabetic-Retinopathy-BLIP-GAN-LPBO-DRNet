"""
Evaluation DTOs: Data Transfer Objects for model evaluation workflows.
SRP — Defines data transfer structures between layers without business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class EvaluationRequestDTO:
    """Model evaluation request parameters."""
    checkpoint_path: str
    config_path: str
    data_mode: Optional[str] = None
    split: str = "test"
    device: Optional[str] = None


@dataclass(frozen=True)
class MetricSummaryDTO:
    """Core computed evaluation metrics summary."""
    accuracy: float
    qwk: float
    macro_auc: Optional[float]
    per_class_auc: Dict[str, float] = field(default_factory=dict)
    classification_report: Dict[str, Any] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)


@dataclass
class EvaluationResponseDTO:
    """Complete response object returned after model evaluation."""
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    metrics: MetricSummaryDTO
    data_mode: str
    split: str
    num_samples: int
