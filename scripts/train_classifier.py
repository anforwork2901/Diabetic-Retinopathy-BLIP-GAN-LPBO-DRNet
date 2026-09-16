"""
Script: Train DR classification model via Clean Architecture pipeline
Usage:
    python scripts/train_classifier.py --config configs/training/rs_27_best.yaml
                                       [--data_mode aptos_gan]
                                       [--epochs 40]
                                       [--batch_size 16]
                                       [--device cuda]
                                       [--output_dir checkpoints/run_rs27]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import yaml

from src.adapters.datasets.aptos_loader import (
    APTOSLoader,
    MessidorLoader,
    build_dataloader,
    build_transforms,
)
from src.adapters.gateways.checkpoint_gateway import CheckpointGateway
from src.adapters.presenters.metrics_presenter import MetricsPresenter
from src.application.dtos.training_dto import TrainingConfigDTO
from src.application.use_cases.evaluate_classifier_use_case import EvaluateClassifierUseCase
from src.application.use_cases.train_classifier_use_case import TrainClassifierUseCase
from src.infrastructure.deep_learning.losses.boundary_focal import (
    AdjacentBoundaryFocalLoss,
    CombinedOrdinalLoss,
)
from src.infrastructure.deep_learning.models.lpbo_drnet.model import (
    LPBOBoundaryConditionedNet,
)
from src.infrastructure.deep_learning.optim.ema import ModelEMA
from src.infrastructure.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LPBO-DRNet via Clean Architecture")
    p.add_argument(
        "--config",
        type=str,
        default="configs/training/rs_27_best.yaml",
        help="Path to YAML configuration file",
    )
    p.add_argument("--data_mode", type=str, default=None, help="Override data_mode in YAML")
    p.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--device", type=str, default=None, help="Device: cuda | cpu | mps")
    p.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load configuration file ───────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_mode = args.data_mode or cfg.get("data_mode", "aptos_gan")
    num_epochs = args.epochs or cfg.get("max_epochs") or cfg.get("num_epochs", 40)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    output_dir = args.output_dir or cfg.get("output_dir") or cfg.get("checkpoint_dir", "checkpoints/run_rs27")
    seed = cfg.get("seed", 42)

    seed_everything(seed)

    # ── 2. Determine Device ──────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using device: {device}")

    # ── 3. Prepare DataLoaders ──────────────────────────────────────────────
    print(f"[Data] Data mode: {data_mode}")
    if data_mode.startswith("aptos"):
        loader_cls = APTOSLoader(
            root_candidates=[Path(p) for p in cfg.get("aptos_root_candidates", [])],
            gan_root_candidates=(
                [Path(p) for p in cfg.get("aptos_gan_root_candidates", [])]
                if "gan" in data_mode
                else None
            ),
            seed=seed,
            num_classes=cfg.get("num_classes", 5),
        )
    else:
        loader_cls = MessidorLoader(
            root_candidates=[Path(p) for p in cfg.get("messidor_root_candidates", [])],
            gan_root_candidates=(
                [Path(p) for p in cfg.get("messidor_gan_root_candidates", [])]
                if "gan" in data_mode
                else None
            ),
            seed=seed,
        )

    img_size = cfg.get("image_size", 320)
    transforms_dict = build_transforms(image_size=img_size, mode="train")

    train_df = loader_cls.get_split("train")
    val_df = loader_cls.get_split("val")
    test_df = loader_cls.get_split("test")

    num_workers = cfg.get("num_workers", 2)
    train_loader = build_dataloader(
        train_df, transforms_dict, mode="train", batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    val_loader = build_dataloader(
        val_df, transforms_dict, mode="val", batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    test_loader = build_dataloader(
        test_df, transforms_dict, mode="val", batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    print(
        f"[Data] Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
    )

    # ── 4. Initialize Model ────────────────────────────────────────────────────
    m_cfg = cfg.get("model", {})
    model = LPBOBoundaryConditionedNet(
        num_classes=cfg.get("num_classes", 5),
        proj_dim=m_cfg.get("proj_dim", 256),
        num_super_tokens=m_cfg.get("num_super_tokens", 16),
        num_heads=m_cfg.get("num_heads", 8),
        sa_layers=m_cfg.get("sa_layers", 2),
        dropout=m_cfg.get("head_dropout", 0.1),
        drop_path_rate=m_cfg.get("drop_path_rate", 0.1),
    ).to(device)

    # ── 5. Initialize Losses ───────────────────────────────────────────────────
    loss_cfg = cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        loss_cfg = {}
    focal_gamma = float(loss_cfg.get("boundary_focal_gamma") or cfg.get("boundary_focal_gamma", 2.0))
    pair_weights = loss_cfg.get("boundary_pair_weights") or cfg.get("pair_weights", [2.0, 2.5, 2.0, 1.5])
    boundary_loss_weight = float(loss_cfg.get("boundary_loss_weight") or cfg.get("boundary_loss_weight", 0.40))
    ce_aux_weight = float(loss_cfg.get("ce_aux_weight") or cfg.get("ce_aux_weight", 0.12))
    prototype_aux_weight = float(loss_cfg.get("prototype_aux_weight") or cfg.get("prototype_aux_weight", 0.08))
    ordinal_bce_weight = float(loss_cfg.get("ordinal_bce_weight") or cfg.get("ordinal_bce_weight", 0.5))

    ordinal_criterion = CombinedOrdinalLoss(
        gamma=focal_gamma,
        ordinal_bce_weight=ordinal_bce_weight,
    )
    boundary_criterion = AdjacentBoundaryFocalLoss(
        gamma=focal_gamma,
        pair_weights=pair_weights,
    )

    # ── 6. Optimizer, LR Scheduler, EMA ───────────────────────────────────────
    lr_backbone = float(cfg.get("lr_backbone", 5e-5))
    lr_head = float(cfg.get("lr_head", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": lr_backbone},
            {"params": model.head_parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    ema_cfg = cfg.get("ema", {})
    if isinstance(ema_cfg, dict):
        ema_enabled = ema_cfg.get("enabled", True)
        ema_decay = float(ema_cfg.get("decay", 0.9995)) if ema_enabled else 0.0
    else:
        ema_decay = float(cfg.get("ema_decay", 0.9995))

    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    patience = int(cfg.get("early_stopping_patience") or cfg.get("patience", 15))

    # ── 7. Training via Use Case ───────────────────────────────────────────
    training_dto = TrainingConfigDTO(
        model_name="LPBO-DRNet",
        num_classes=cfg.get("num_classes", 5),
        data_mode=data_mode,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        boundary_loss_weight=boundary_loss_weight,
        boundary_focal_gamma=focal_gamma,
        ce_aux_weight=ce_aux_weight,
        prototype_aux_weight=prototype_aux_weight,
        patience=patience,
        seed=seed,
        output_dir=output_dir,
    )

    def epoch_callback(m):
        star_acc = " ★ Best ACC" if m.is_best_acc else ""
        star_qwk = " ★ Best QWK" if m.is_best_qwk else ""
        print(
            f"Epoch {m.epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {m.train_loss:.4f} Acc: {m.train_acc*100:.2f}% | "
            f"Val Loss: {m.val_loss:.4f} Acc: {m.val_acc*100:.2f}% QWK: {m.val_qwk:.4f}"
            f"{star_acc}{star_qwk}"
        )

    print("\n" + "=" * 70)
    print(" STARTING LPBO-DRNET TRAINING (CLEAN ARCHITECTURE)")
    print("=" * 70)

    checkpoint_gateway = CheckpointGateway(output_dir=Path(output_dir))
    train_use_case = TrainClassifierUseCase(checkpoint_gateway=checkpoint_gateway)
    history = train_use_case.execute(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ordinal_criterion=ordinal_criterion,
        boundary_criterion=boundary_criterion,
        cfg=training_dto,
        device=device,
        ema=ema,
        on_epoch_end_callback=epoch_callback,
    )

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print(f" Best Val ACC: {history.best_val_acc*100:.2f}% | Best Val QWK: {history.best_val_qwk:.4f}")
    print(f" Checkpoints saved at: {output_dir}")
    print("=" * 70)

    # ── 8. Independent Evaluation on Test Set ───────────────────────────
    if len(test_df) > 0:
        print("\n[Evaluation] Evaluating Test Set with best checkpoint...")
        primary_ckpt = Path(output_dir) / "selected_primary.pt"
        if primary_ckpt.exists():
            gateway = CheckpointGateway()
            gateway.load_model_state(model, primary_ckpt, map_location=device)

        eval_use_case = EvaluateClassifierUseCase()
        res = eval_use_case.execute(
            model=model, loader=test_loader, device=device, data_mode=data_mode, split="test"
        )
        print(f"\n[Test Results] Accuracy: {res.metrics.accuracy * 100:.2f}% | QWK: {res.metrics.qwk:.4f}")
        if res.metrics.macro_auc is not None:
            print(f"[Test Results] Macro ROC-AUC: {res.metrics.macro_auc:.4f}")


if __name__ == "__main__":
    main()
