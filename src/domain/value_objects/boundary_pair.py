"""
Domain Value Objects: BoundaryPair & EvaluationMetrics
SRP — Encapsulates immutable values related to adjacent boundary pairs
      and aggregated model evaluation metrics.
      Completely decoupled from any ML/DL framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundaryPair:
    """
    Pair of adjacent DR grades forming a clinical decision boundary.

    Example: BoundaryPair(0, 1) = boundary No DR ↔ Mild.
    """

    lower: int   # Lower grade
    upper: int   # Upper grade (= lower + 1)

    def __post_init__(self) -> None:
        if self.upper != self.lower + 1:
            raise ValueError(
                f"BoundaryPair must consist of adjacent grades: upper={self.upper}, lower={self.lower}"
            )

    @property
    def index(self) -> int:
        """0-based index of the boundary pair in ordered lists."""
        return self.lower

    @property
    def description(self) -> str:
        return f"Grade{self.lower}↔Grade{self.upper}"

    @classmethod
    def all_pairs(cls, num_classes: int = 5) -> List["BoundaryPair"]:
        """Returns all contiguous boundary pairs for num_classes."""
        return [cls(k, k + 1) for k in range(num_classes - 1)]


@dataclass
class EvaluationMetrics:
    """
    Aggregated evaluation metrics on a test or validation split.

    Attributes
    ----------
    accuracy : float       Accuracy score (0.0 to 1.0).
    qwk : float            Quadratic Weighted Kappa score.
    auc : float            Macro-averaged ROC-AUC score.
    per_class_auc : Dict   Per-class ROC-AUC scores.
    num_correct : int      Count of correctly classified samples.
    num_total : int        Total number of samples evaluated.
    """

    accuracy: float
    qwk: float
    auc: float
    per_class_auc: Dict[int, float]
    num_correct: int
    num_total: int

    @property
    def num_wrong(self) -> int:
        return self.num_total - self.num_correct

    def to_dict(self) -> Dict[str, float]:
        """Converts to a flat dictionary for CSV / LaTeX / JSON serialization."""
        d: Dict[str, float] = {
            "accuracy": round(self.accuracy * 100, 4),
            "qwk": round(self.qwk * 100, 4),
            "auc": round(self.auc * 100, 4),
            "num_correct": self.num_correct,
            "num_total": self.num_total,
        }
        for cls_idx, cls_auc in self.per_class_auc.items():
            d[f"auc_grade_{cls_idx}"] = round(cls_auc * 100, 4)
        return d

    def summary(self) -> str:
        """Formatted multi-line summary string for console logging."""
        lines = [
            f"  Accuracy : {self.accuracy * 100:.2f}%  "
            f"({self.num_correct}/{self.num_total})",
            f"  QW Kappa : {self.qwk * 100:.4f}%",
            f"  ROC-AUC  : {self.auc * 100:.4f}%",
        ]
        for cls_idx, cls_auc in self.per_class_auc.items():
            lines.append(f"    Grade {cls_idx} AUC: {cls_auc:.4f}")
        return "\n".join(lines)
