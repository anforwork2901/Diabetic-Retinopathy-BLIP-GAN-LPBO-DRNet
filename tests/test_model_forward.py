"""
Smoke Test: Verify forward pass and tensor shapes of LPBO-DRNet
Run: python tests/test_model_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.infrastructure.deep_learning.models.lpbo_drnet.model import (
    LPBOBoundaryConditionedNet,
)
from src.infrastructure.deep_learning.losses.boundary_focal import (
    AdjacentBoundaryFocalLoss, CombinedOrdinalLoss,
)
from src.infrastructure.deep_learning.optim.ema import EMA
from src.infrastructure.image_processing.letterbox import LetterboxPad
from src.domain.entities.dr_grade import DRGrade
from src.domain.value_objects.boundary_pair import BoundaryPair
from src.infrastructure.utils.seed import seed_everything


def test_dr_grade():
    print("\n[1] Test DRGrade entity...")
    assert DRGrade.NO_DR.value == 0
    assert DRGrade.PROLIFERATIVE.display_name == "Proliferative DR"
    assert DRGrade.MODERATE.is_referable is True
    assert DRGrade.num_classes() == 5
    print("    ✅ DRGrade: OK")


def test_boundary_pairs():
    print("\n[2] Test BoundaryPair value object...")
    pairs = BoundaryPair.all_pairs(5)
    assert len(pairs) == 4
    assert pairs[1].description == "Grade1↔Grade2"
    try:
        BoundaryPair(0, 2)  # Must raise ValueError
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("    ✅ BoundaryPair: OK")


def test_letterbox():
    print("\n[3] Test LetterboxPad...")
    from PIL import Image
    img = Image.new("RGB", (400, 300))  # Landscape
    pad = LetterboxPad(320)
    result = pad(img)
    assert result.size == (320, 320), f"Expected (320,320), got {result.size}"
    print("    ✅ LetterboxPad: OK")


def test_losses():
    print("\n[4] Test Loss Functions...")
    seed_everything(42)
    B, K = 8, 5  # Batch=8, 5 classes

    # OrdinalBCE + CombinedOrdinalLoss
    binary_logits = torch.randn(B, K - 1)
    targets = torch.randint(0, K, (B,))

    from src.infrastructure.deep_learning.models.lpbo_drnet.blocks import OrdinalHead
    ordinal_head = OrdinalHead(16, K)

    combined = CombinedOrdinalLoss(gamma=2.0)
    feat = torch.randn(B, 16)
    dummy_logits = ordinal_head(feat)
    loss_c = combined(dummy_logits, targets, ordinal_head)
    assert loss_c.item() > 0
    print(f"    CombinedOrdinalLoss: {loss_c.item():.4f} ✅")

    # AdjacentBoundaryFocalLoss
    boundary_logits = torch.randn(B, K - 1)
    boundary_loss = AdjacentBoundaryFocalLoss(
        pair_weights=[2.00, 2.50, 2.00, 1.50], gamma=2.0
    )
    loss_b = boundary_loss(boundary_logits, targets)
    assert loss_b.item() > 0
    print(f"    AdjacentBoundaryFocalLoss: {loss_b.item():.4f} ✅")


def test_model_forward():
    print("\n[5] Test LPBO-DRNet forward pass...")
    seed_everything(42)
    device = torch.device("cpu")

    model = LPBOBoundaryConditionedNet(
        num_classes=5, proj_dim=64,     # Smaller proj_dim for faster testing
        num_super_tokens=4, num_heads=4,
        sa_layers=1, dropout=0.0, drop_path_rate=0.0,
    ).to(device)

    # Dummy batch of 2 images 3x320x320
    dummy = torch.zeros(2, 3, 320, 320, device=device)

    # Standard forward pass
    out = model(dummy, return_boundary=False)
    # Ordinal logits: (B, num_classes-1) = (2, 4)
    assert out.shape == (2, 4), f"Expected (2,4), got {out.shape}"
    print(f"    Ordinal logits shape: {out.shape} ✅")

    # Forward pass with boundary outputs
    out_full = model(dummy, return_boundary=True)
    assert "ordinal_logits" in out_full
    assert "boundary_logits" in out_full
    assert "ce_logits" in out_full
    assert "prototype_logits" in out_full
    assert out_full["ordinal_logits"].shape == (2, 4)
    assert out_full["boundary_logits"].shape == (2, 4)
    assert out_full["ce_logits"].shape == (2, 5)
    assert out_full["prototype_logits"].shape == (2, 5)
    print(f"    boundary_logits shape: {out_full['boundary_logits'].shape} ✅")
    print(f"    ce_logits shape:       {out_full['ce_logits'].shape} ✅")
    print(f"    prototype_logits:      {out_full['prototype_logits'].shape} ✅")

    # predict_proba
    probs = model.predict_proba(dummy)
    assert probs.shape == (2, 5)
    assert abs(probs.sum(dim=1).mean().item() - 1.0) < 1e-5, "Probs must sum to 1"
    print(f"    predict_proba shape: {probs.shape}, sum≈1: ✅")


def test_ema():
    print("\n[6] Test EMA...")
    seed_everything(42)
    from src.infrastructure.deep_learning.models.lpbo_drnet.blocks import OrdinalHead
    model = OrdinalHead(32, 5)
    ema = EMA(model, decay=0.9995)

    # Simulate an update step
    with torch.no_grad():
        for p in model.parameters():
            p.data.fill_(1.0)
    ema.update(model)

    sd = ema.state_dict()
    assert sd is not None
    print("    ✅ EMA: OK")


def main():
    print("=" * 60)
    print("  SMOKE TEST — BLIP-DRNet Clean Architecture")
    print("=" * 60)

    test_dr_grade()
    test_boundary_pairs()
    test_letterbox()
    test_losses()
    test_model_forward()
    test_ema()

    print("\n" + "=" * 60)
    print("  ✅ ALL SMOKE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
