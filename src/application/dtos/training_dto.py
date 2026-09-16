"""
Training DTOs: Data Transfer Objects for model training workflows.
SRP — Defines data transfer structures without business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TrainingConfigDTO:
    """Training configuration parameters parsed from YAML/CLI."""
    model_name: str
    num_classes: int
    data_mode: str
    batch_size: int
    num_epochs: int
    lr_backbone: float
    lr_head: float
    weight_decay: float
    ema_decay: float
    boundary_loss_weight: float
    boundary_focal_gamma: float
    ce_aux_weight: float
    prototype_aux_weight: float
    patience: int
    seed: int
    output_dir: str


@dataclass(frozen=True)
class EpochMetricsDTO:
    """Evaluation metrics recorded after each training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_qwk: float
    lr_backbone: float
    lr_head: float
    is_best_acc: bool = False
    is_best_qwk: bool = False
    is_best_loss: bool = False


@dataclass
class TrainingHistoryDTO:
    """Full historical metrics and checkpoint paths across training epochs."""
    epochs: List[EpochMetricsDTO] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    best_val_qwk: float = 0.0
    best_val_loss: float = float("inf")
    checkpoint_paths: Dict[str, str] = field(default_factory=dict)
