"""
Script: Evaluate model from checkpoint
Usage: python scripts/evaluate.py --config configs/training/rs_27_best.yaml
                                  --checkpoint checkpoints/best/selected_primary.pt
                                  [--data_mode aptos_gan]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to import src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.adapters.datasets.aptos_loader import (
    APTOSLoader, MessidorLoader, build_transforms, build_dataloader,
)
from src.adapters.gateways.checkpoint_gateway import CheckpointGateway
from src.adapters.presenters.metrics_presenter import MetricsCalculator, MetricsPresenter
from src.infrastructure.deep_learning.models.lpbo_drnet.model import (
    LPBOBoundaryConditionedNet,
)
from src.infrastructure.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LPBO-DRNet from checkpoint")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file (e.g., configs/training/rs_27_best.yaml)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--data_mode", type=str, default=None,
                   help="Override data_mode in config (aptos_only|aptos_gan|messidor_only|messidor_gan)")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: cuda | cpu | mps (default: auto-detect)")
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: LPBOBoundaryConditionedNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference across the entire loader, returning (y_true, y_pred, y_prob)."""
    model.eval()
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        probs = model.predict_proba(imgs)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = y_prob.argmax(1)
    return y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_mode = args.data_mode or cfg["data_mode"]
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print(f"Data mode: {data_mode}")
    if data_mode.startswith("aptos"):
        loader_cls = APTOSLoader(
            root_candidates=[Path(p) for p in cfg.get("aptos_root_candidates", [])],
            gan_root_candidates=(
                [Path(p) for p in cfg.get("aptos_gan_root_candidates", [])]
                if "gan" in data_mode else None
            ),
            seed=seed,
            num_classes=cfg.get("num_classes", 5),
        )
    else:
        loader_cls = MessidorLoader(
            root_candidates=[Path(p) for p in cfg.get("messidor_root_candidates", [])],
            gan_root_candidates=(
                [Path(p) for p in cfg.get("messidor_gan_root_candidates", [])]
                if "gan" in data_mode else None
            ),
            seed=seed,
        )

    test_df = loader_cls.get_split(args.split)
    img_size = cfg.get("image_size", 320)
    transforms_dict = build_transforms(img_size, mode="val")
    test_loader = build_dataloader(
        test_df, transforms_dict, mode="val",
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 2),
        seed=seed,
    )
    print(f"Split '{args.split}': {len(test_df)} images")

    # ── Load Model ───────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = LPBOBoundaryConditionedNet(
        num_classes=cfg["num_classes"],
        proj_dim=model_cfg["proj_dim"],
        num_super_tokens=model_cfg["num_super_tokens"],
        num_heads=model_cfg["num_heads"],
        sa_layers=model_cfg["sa_layers"],
        dropout=model_cfg["head_dropout"],
        drop_path_rate=model_cfg["drop_path_rate"],
    ).to(device)

    gw = CheckpointGateway(output_dir=Path(args.checkpoint).parent)
    ckpt_info = gw.load_model_state(
        model, Path(args.checkpoint), map_location=device, use_ema=True
    )
    print(f"Checkpoint loaded: {Path(args.checkpoint).name}")
    if "best_val_acc" in ckpt_info:
        print(f"  Val acc at checkpoint: {ckpt_info['best_val_acc']*100:.2f}%")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)

    calculator = MetricsCalculator(num_classes=cfg["num_classes"])
    metrics = calculator.compute(y_true, y_pred, y_prob)

    MetricsPresenter.print_summary(
        f"LPBO-DRNet [{data_mode}] — {args.split.upper()} RESULTS", metrics
    )

    print("\nClassification Report:")
    print(calculator.classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
