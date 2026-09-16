"""
Domain Entity: PredictionResult
SRP — Encapsulates classification prediction results from models.
      Completely decoupled from any ML/DL framework.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .dr_grade import DRGrade


@dataclass
class PredictionResult:
    """
    Encapsulates Diabetic Retinopathy prediction results for a fundus photograph.

    Attributes
    ----------
    predicted_grade : DRGrade
        Predicted DR clinical grade.
    class_probabilities : List[float]
        Class probability distribution [P(G0), P(G1), P(G2), P(G3), P(G4)].
    boundary_scores : Optional[List[float]]
        Scores across adjacent boundaries [B(0-1), B(1-2), B(2-3), B(3-4)].
    true_grade : Optional[DRGrade]
        Ground-truth grade if available (used during evaluation).
    """

    predicted_grade: DRGrade
    class_probabilities: List[float]
    boundary_scores: Optional[List[float]] = None
    true_grade: Optional[DRGrade] = None

    @property
    def confidence(self) -> float:
        """Prediction confidence (probability of the chosen class)."""
        return float(self.class_probabilities[int(self.predicted_grade)])

    @property
    def expected_rank(self) -> float:
        """
        Expected Rank — used for Boundary-Aware Ordinal (BAO) post-processing.
        ER = sum(P(Gi) * i) for i in [0..4]
        """
        probs = np.array(self.class_probabilities, dtype=np.float32)
        ranks = np.arange(len(probs), dtype=np.float32)
        return float((probs * ranks).sum())

    @property
    def is_correct(self) -> Optional[bool]:
        """Compares prediction against ground truth. Returns None if true_grade is unset."""
        if self.true_grade is None:
            return None
        return self.predicted_grade == self.true_grade

    @property
    def grade_name(self) -> str:
        """Human-readable display name of the predicted DR grade."""
        return self.predicted_grade.display_name
