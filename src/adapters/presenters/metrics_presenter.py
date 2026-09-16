"""
Metrics Presenter: Evaluation Metric Computation and Presentation
SRP — Solely responsible for metric calculation and output formatting,
      decoupled from model training or weight loading logic.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.domain.value_objects.boundary_pair import EvaluationMetrics


class MetricsCalculator:
    """
    Computes comprehensive evaluation metrics for Diabetic Retinopathy grading.

    Supported metrics: Accuracy, Quadratic Weighted Kappa (QWK), macro ROC-AUC,
    per-class AUC, and scikit-learn classification report.
    """

    def __init__(self, num_classes: int = 5) -> None:
        self.num_classes = num_classes
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> EvaluationMetrics:
        """
        Computes comprehensive evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray (N,)     Ground-truth integer labels.
        y_pred : np.ndarray (N,)     Predicted class labels (argmax).
        y_prob : np.ndarray (N, K)   Predicted class probabilities.

        Returns
        -------
        EvaluationMetrics
        """
        acc = accuracy_score(y_true, y_pred)
        num_correct = int(np.sum(y_true == y_pred))
        num_total = len(y_true)

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        macro_auc = self._compute_macro_auc(y_true, y_prob)
        per_class_auc = self._per_class_auc(y_true, y_prob)

        return EvaluationMetrics(
            accuracy=acc,
            qwk=qwk,
            auc=macro_auc,
            per_class_auc=per_class_auc,
            num_correct=num_correct,
            num_total=num_total,
        )

    def classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """Returns formatted classification report string."""
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names[: self.num_classes],
            digits=4,
        )

    def _compute_macro_auc(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> float:
        try:
            y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            return float(
                roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            return float("nan")

    def _per_class_auc(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[int, float]:
        y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        aucs: Dict[int, float] = {}
        for cls_idx in range(self.num_classes):
            if len(np.unique(y_bin[:, cls_idx])) < 2:
                aucs[cls_idx] = float("nan")
            else:
                aucs[cls_idx] = float(
                    roc_auc_score(y_bin[:, cls_idx], y_prob[:, cls_idx])
                )
        return aucs


class MetricsPresenter:
    """
    Formats and exports EvaluationMetrics to console, tables, and LaTeX.
    SRP — Exclusively handles formatting and presentation, not calculation.
    """

    @staticmethod
    def print_summary(title: str, metrics: EvaluationMetrics) -> None:
        """Prints summary report to console."""
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
        print(metrics.summary())

    @staticmethod
    def to_dataframe(metrics_by_mode: Dict[str, EvaluationMetrics]) -> pd.DataFrame:
        """
        Converts dict {mode_name: EvaluationMetrics} into a comparison DataFrame.
        """
        rows = []
        for mode, m in metrics_by_mode.items():
            row = {"mode": mode}
            row.update(m.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def to_latex_row(metrics: EvaluationMetrics) -> str:
        """
        Generates a LaTeX table row for research publication tables.

        Example: & 98.15 & 99.06 & 99.08 \\\\
        """
        return (
            f"& {metrics.accuracy * 100:.2f} "
            f"& {metrics.qwk * 100:.2f} "
            f"& {metrics.auc * 100:.2f} \\\\"
        )

    @staticmethod
    def print_four_modes_table(
        modes: List[str], metrics_list: List[EvaluationMetrics]
    ) -> None:
        """Prints comparative 4-mode benchmark table to console."""
        header = f"{'Mode':<30} {'ACC(%)':>8} {'QWK(%)':>8} {'AUC(%)':>8}"
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for mode, m in zip(modes, metrics_list):
            print(
                f"{mode:<30} "
                f"{m.accuracy * 100:>8.2f} "
                f"{m.qwk * 100:>8.4f} "
                f"{m.auc * 100:>8.4f}"
            )
        print("=" * len(header))
