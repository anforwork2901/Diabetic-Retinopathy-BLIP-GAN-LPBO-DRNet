"""
Evaluate Classifier Use Case
SRP — Orchestrates the full evaluation pipeline for a checkpointed model.
DIP — Injects ClassifierInterface, CheckpointGateway, and DataLoaders via Dependency Injection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.adapters.gateways.checkpoint_gateway import CheckpointGateway
from src.adapters.presenters.metrics_presenter import MetricsCalculator
from src.application.dtos.evaluation_dto import (
    EvaluationRequestDTO,
    EvaluationResponseDTO,
    MetricSummaryDTO,
)
from src.domain.interfaces.classifier_interface import ClassifierInterface


class EvaluateClassifierUseCase:
    """Use case orchestrating Diabetic Retinopathy classification evaluation."""

    def __init__(
        self,
        checkpoint_gateway: Optional[CheckpointGateway] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ) -> None:
        self.checkpoint_gateway = checkpoint_gateway or CheckpointGateway()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

    @torch.no_grad()
    def run_inference(
        self,
        model: ClassifierInterface,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Executes forward inference over loader, returning (y_true, y_pred, y_prob)."""
        if hasattr(model, "eval"):
            model.eval()

        all_labels, all_probs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = model.predict_proba(imgs)
            if hasattr(probs, "detach"):
                probs = probs.detach()
            if hasattr(probs, "cpu"):
                probs = probs.cpu().numpy()
            all_probs.extend(probs)
            if hasattr(labels, "cpu"):
                labels = labels.cpu().tolist()
            all_labels.extend(labels)

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = y_prob.argmax(axis=1)
        return y_true, y_pred, y_prob

    def execute(
        self,
        model: ClassifierInterface,
        loader: DataLoader,
        device: torch.device,
        data_mode: str = "aptos_gan",
        split: str = "test",
    ) -> EvaluationResponseDTO:
        """Executes evaluation pipeline and returns complete EvaluationResponseDTO."""
        y_true, y_pred, y_prob = self.run_inference(model, loader, device)

        eval_metrics = self.metrics_calculator.compute(y_true, y_pred, y_prob)

        summary = MetricSummaryDTO(
            accuracy=eval_metrics.accuracy,
            qwk=eval_metrics.qwk,
            macro_auc=eval_metrics.auc,
            per_class_auc={str(k): v for k, v in eval_metrics.per_class_auc.items()},
            classification_report={},
            confusion_matrix=[],
        )

        return EvaluationResponseDTO(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metrics=summary,
            data_mode=data_mode,
            split=split,
            num_samples=len(y_true),
        )
