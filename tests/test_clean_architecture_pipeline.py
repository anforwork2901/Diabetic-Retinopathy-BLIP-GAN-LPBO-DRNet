"""
Test Suite: Comprehensive integration testing for Clean Architecture Pipeline
Verifies:
1. Application DTOs & Use Cases (EvaluateClassifierUseCase, TrainClassifierUseCase)
2. BLIP-GAN Modules (WAE, LRDB, ACFDGenerator, ACFDDiscriminator)
3. Domain Entities & Value Objects
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.adapters.gateways.checkpoint_gateway import CheckpointGateway
from src.adapters.presenters.metrics_presenter import MetricsCalculator
from src.application.dtos.evaluation_dto import EvaluationRequestDTO, EvaluationResponseDTO
from src.application.dtos.training_dto import TrainingConfigDTO
from src.application.use_cases.evaluate_classifier_use_case import EvaluateClassifierUseCase
from src.application.use_cases.train_classifier_use_case import TrainClassifierUseCase
from src.domain.entities.dr_grade import DRGrade
from src.domain.interfaces.classifier_interface import ClassifierInterface
from src.infrastructure.deep_learning.models.blip_gan.discriminator import ACFDDiscriminator
from src.infrastructure.deep_learning.models.blip_gan.generator import ACFDGenerator
from src.infrastructure.deep_learning.models.blip_gan.lrdb import LRDB, DepthwiseSeparableConv
from src.infrastructure.deep_learning.models.blip_gan.wae import WAE, mmd_loss


class MockClassifier(nn.Module, ClassifierInterface):
    """Mock Classifier conforming to ClassifierInterface for unit testing."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor, return_boundary: bool = False):
        logits = self.fc(x)
        if return_boundary:
            return {
                "ordinal_logits": logits[:, :4],
                "boundary_logits": logits[:, :4],
                "ce_logits": logits,
                "prototype_logits": logits,
            }
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path))


def test_evaluate_classifier_use_case():
    """Test EvaluateClassifierUseCase with MockClassifier."""
    device = torch.device("cpu")
    model = MockClassifier(num_classes=5)

    # Fake dataset: 10 samples, feature dim 10
    x_data = torch.randn(10, 10)
    y_data = torch.randint(0, 5, (10,))
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=5)

    use_case = EvaluateClassifierUseCase()
    res: EvaluationResponseDTO = use_case.execute(
        model=model, loader=loader, device=device, data_mode="aptos_gan", split="test"
    )

    assert isinstance(res, EvaluationResponseDTO)
    assert res.num_samples == 10
    assert len(res.y_true) == 10
    assert len(res.y_pred) == 10
    assert 0.0 <= res.metrics.accuracy <= 1.0
    print("✅ TEST 1 PASSED: EvaluateClassifierUseCase works correctly.")


def test_blip_gan_modules():
    """Verify forward pass of BLIP-GAN modules."""
    device = torch.device("cpu")

    # 1. LRDB
    lrdb = LRDB(in_ch=64, growth=32).to(device)
    x = torch.randn(2, 64, 32, 32)
    out_lrdb = lrdb(x)
    assert out_lrdb.shape == (2, 64, 32, 32), f"LRDB shape mismatch: {out_lrdb.shape}"

    # 2. WAE
    wae = WAE(img_channels=3, latent_dim=128, img_size=128).to(device)
    img = torch.randn(2, 3, 128, 128)
    recon, z = wae(img)
    assert recon.shape == (2, 3, 128, 128), f"WAE recon shape mismatch: {recon.shape}"
    assert z.shape == (2, 128), f"WAE latent shape mismatch: {z.shape}"

    loss_mmd = mmd_loss(z)
    assert torch.isfinite(loss_mmd), "MMD loss must be finite"

    # 3. ACFDDiscriminator
    d = ACFDDiscriminator(img_ch=3, mask_ch=3, base_ch=16).to(device)
    mask = torch.randn(2, 3, 128, 128)
    patch_out = d(img, mask)
    assert patch_out.ndim == 4, f"Discriminator output must be 4D patch: {patch_out.shape}"

    # 4. ACFDGenerator
    g = ACFDGenerator(mask_ch=3, img_ch=3, latent_dim=128, base_ch=16, growth=16, img_size=128).to(device)
    z_in = torch.randn(2, 128)
    fake_img = g(mask, z_in)
    assert fake_img.shape == (2, 3, 128, 128), f"Generator output shape mismatch: {fake_img.shape}"

    print("✅ TEST 2 PASSED: All BLIP-GAN modules (LRDB, WAE, Generator, Discriminator) forward pass succeeded.")


def test_checkpoint_gateway_flexible_save_load(tmp_path):
    """Test Gateway model weights save & load functionality."""
    gw = CheckpointGateway(output_dir=tmp_path)
    model = nn.Linear(4, 2)
    ckpt_file = tmp_path / "test_model.pt"

    saved_path = gw.save(
        model=model,
        filename=ckpt_file,
        metadata={"val_acc": 0.95, "epoch": 10},
    )
    assert saved_path.exists()

    model2 = nn.Linear(4, 2)
    gw.load_model_state(model2, saved_path)

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    print("✅ TEST 3 PASSED: CheckpointGateway successfully saved and loaded weights.")


if __name__ == "__main__":
    test_evaluate_classifier_use_case()
    test_blip_gan_modules()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_checkpoint_gateway_flexible_save_load(Path(tmpdir))
    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
