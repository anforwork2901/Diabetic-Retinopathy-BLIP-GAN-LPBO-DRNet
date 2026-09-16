"""
Train Classifier Use Case
SRP — Orchestrates the end-to-end training cycle of the DR classification model.
DIP — Receives Model, Loss, Optimizer, Gateways, and Dataloaders via Dependency Injection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader

from src.adapters.gateways.checkpoint_gateway import CheckpointGateway
from src.application.dtos.training_dto import (
    EpochMetricsDTO,
    TrainingConfigDTO,
    TrainingHistoryDTO,
)
from src.infrastructure.deep_learning.optim.ema import ModelEMA

logger = logging.getLogger(__name__)


class TrainClassifierUseCase:
    """Use case orchestrating LPBO-DRNet model training."""

    def __init__(
        self,
        checkpoint_gateway: Optional[CheckpointGateway] = None,
    ) -> None:
        self.checkpoint_gateway = checkpoint_gateway or CheckpointGateway()

    def compute_total_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        ordinal_criterion: nn.Module,
        boundary_criterion: nn.Module,
        ordinal_head: nn.Module,
        boundary_weight: float = 1.0,
        ce_aux_weight: float = 0.2,
        prototype_aux_weight: float = 0.1,
    ) -> torch.Tensor:
        """Computes multi-task composite loss."""
        ordinal_logits = outputs["ordinal_logits"]
        boundary_logits = outputs["boundary_logits"]

        # 1. Ordinal Loss
        ord_loss = ordinal_criterion(ordinal_logits, labels, ordinal_head)

        # 2. Boundary Focal Loss
        bnd_loss = boundary_criterion(boundary_logits, labels)

        # 3. CE Loss
        ce_loss = F.cross_entropy(outputs["ce_logits"], labels)

        # 4. Prototype Aux Loss
        proto_loss = F.cross_entropy(outputs["prototype_logits"], labels)

        return (
            ord_loss
            + boundary_weight * bnd_loss
            + ce_aux_weight * ce_loss
            + prototype_aux_weight * proto_loss
        )

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        ordinal_criterion: nn.Module,
        boundary_criterion: nn.Module,
        device: torch.device,
        ema: Optional[ModelEMA] = None,
        grad_clip_norm: float = 1.0,
        boundary_weight: float = 1.0,
        ce_aux_weight: float = 0.2,
        prototype_aux_weight: float = 0.1,
        use_amp: bool = True,
    ) -> Tuple[float, float]:
        """Trains for one epoch, returning (train_loss, train_acc)."""
        model.train()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=(use_amp and device.type == "cuda")):
                outputs = model(imgs, return_boundary=True)
                loss = self.compute_total_loss(
                    outputs=outputs,
                    labels=labels,
                    ordinal_criterion=ordinal_criterion,
                    boundary_criterion=boundary_criterion,
                    ordinal_head=model.head.ordinal_head,
                    boundary_weight=boundary_weight,
                    ce_aux_weight=ce_aux_weight,
                    prototype_aux_weight=prototype_aux_weight,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            total_loss += loss.item() * imgs.size(0)

            with torch.no_grad():
                probs = model.head.ordinal_head.logits_to_proba(outputs["ordinal_logits"].detach())
                preds = probs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return float(epoch_loss), float(epoch_acc)

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        ordinal_criterion: nn.Module,
        boundary_criterion: nn.Module,
        device: torch.device,
        boundary_weight: float = 1.0,
        ce_aux_weight: float = 0.2,
        prototype_aux_weight: float = 0.1,
    ) -> Tuple[float, float, float]:
        """Evaluates on the validation set, returning (val_loss, val_acc, val_qwk)."""
        model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs, return_boundary=True)
            loss = self.compute_total_loss(
                outputs=outputs,
                labels=labels,
                ordinal_criterion=ordinal_criterion,
                boundary_criterion=boundary_criterion,
                ordinal_head=model.head.ordinal_head,
                boundary_weight=boundary_weight,
                ce_aux_weight=ce_aux_weight,
                prototype_aux_weight=prototype_aux_weight,
            )

            total_loss += loss.item() * imgs.size(0)
            probs = model.head.ordinal_head.logits_to_proba(outputs["ordinal_logits"])
            preds = probs.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        val_loss = total_loss / len(loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        return float(val_loss), float(val_acc), float(val_qwk)

    def execute(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        ordinal_criterion: nn.Module,
        boundary_criterion: nn.Module,
        cfg: TrainingConfigDTO,
        device: torch.device,
        ema: Optional[ModelEMA] = None,
        on_epoch_end_callback: Optional[Callable[[EpochMetricsDTO], None]] = None,
    ) -> TrainingHistoryDTO:
        """Executes the complete multi-epoch training pipeline."""
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        history = TrainingHistoryDTO()
        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        best_val_acc = 0.0
        best_val_qwk = 0.0
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                ordinal_criterion=ordinal_criterion,
                boundary_criterion=boundary_criterion,
                device=device,
                ema=ema,
                boundary_weight=cfg.boundary_loss_weight,
                ce_aux_weight=cfg.ce_aux_weight,
                prototype_aux_weight=cfg.prototype_aux_weight,
                use_amp=use_cuda,
            )

            # Evaluate with EMA shadow weights if available
            if ema is not None:
                backup = ema.apply_to(model)
                val_loss, val_acc, val_qwk = self.validate(
                    model=model,
                    loader=val_loader,
                    ordinal_criterion=ordinal_criterion,
                    boundary_criterion=boundary_criterion,
                    device=device,
                    boundary_weight=cfg.boundary_loss_weight,
                    ce_aux_weight=cfg.ce_aux_weight,
                    prototype_aux_weight=cfg.prototype_aux_weight,
                )
                ema.restore(model, backup)
            else:
                val_loss, val_acc, val_qwk = self.validate(
                    model=model,
                    loader=val_loader,
                    ordinal_criterion=ordinal_criterion,
                    boundary_criterion=boundary_criterion,
                    device=device,
                    boundary_weight=cfg.boundary_loss_weight,
                    ce_aux_weight=cfg.ce_aux_weight,
                    prototype_aux_weight=cfg.prototype_aux_weight,
                )

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Retrieve current learning rates
            lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            lr_b = lrs[0] if len(lrs) > 0 else 0.0
            lr_h = lrs[1] if len(lrs) > 1 else lr_b

            is_best_acc = val_acc > best_val_acc
            is_best_qwk = val_qwk > best_val_qwk
            is_best_loss = val_loss < best_val_loss

            if is_best_acc:
                best_val_acc = val_acc
            if is_best_qwk:
                best_val_qwk = val_qwk

            # Save primary best checkpoint
            if is_best_acc or (val_acc == best_val_acc and val_qwk >= best_val_qwk):
                primary_path = out_dir / "selected_primary.pt"
                self.checkpoint_gateway.save(
                    model=model,
                    filename=primary_path,
                    metadata={
                        "epoch": epoch,
                        "metrics": {"val_acc": val_acc, "val_qwk": val_qwk, "val_loss": val_loss},
                        "data_mode": cfg.data_mode,
                        "config": cfg.__dict__,
                    },
                    ema=ema,
                    optimizer=optimizer,
                )
                history.checkpoint_paths["selected_primary"] = str(primary_path)

            if is_best_loss:
                best_val_loss = val_loss
                patience_counter = 0
                loss_path = out_dir / "best_loss.pt"
                self.checkpoint_gateway.save(
                    model=model,
                    filename=loss_path,
                    metadata={
                        "epoch": epoch,
                        "metrics": {"val_acc": val_acc, "val_qwk": val_qwk, "val_loss": val_loss},
                    },
                    ema=ema,
                    optimizer=optimizer,
                )
                history.checkpoint_paths["best_loss"] = str(loss_path)
            else:
                patience_counter += 1

            epoch_dto = EpochMetricsDTO(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_qwk=val_qwk,
                lr_backbone=lr_b,
                lr_head=lr_h,
                is_best_acc=is_best_acc,
                is_best_qwk=is_best_qwk,
                is_best_loss=is_best_loss,
            )
            history.epochs.append(epoch_dto)

            if on_epoch_end_callback:
                on_epoch_end_callback(epoch_dto)

            # Early stopping check
            if patience_counter >= cfg.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        history.best_val_acc = best_val_acc
        history.best_val_qwk = best_val_qwk
        history.best_val_loss = best_val_loss
        return history
