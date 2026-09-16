"""
Dataset Adapter: APTOS 2019 & Messidor Loaders + PyTorch DRDataset
SRP — Each class handles a specific dataset source.
LSP — APTOSLoader and MessidorLoader implement the same IDatasetLoader interface,
      interchangeable across Use Cases without modification.
Extracted and refactored from research experiment notebooks.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.domain.interfaces.classifier_interface import IDatasetLoader
from src.infrastructure.image_processing.letterbox import LetterboxPad

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".png", ".jpg", ".jpeg"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of an image file for duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_candidate(candidates: List[Path]) -> Optional[Path]:
    """Finds the first existing candidate directory from a priority list."""
    for p in candidates:
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# APTOS 2019 Loader
# ─────────────────────────────────────────────────────────────────────────────
class APTOSLoader(IDatasetLoader):
    """
    Loads APTOS 2019 data, filters duplicate image hashes, and splits 60/20/20.

    Parameters
    ----------
    root_candidates : List[Path]
        Ordered list of candidate root paths to probe.
    gan_root_candidates : Optional[List[Path]]
        Candidate paths for GAN synthetic data to append to the training split.
    seed : int
        Random seed for deterministic reproducibility.
    num_classes : int
        Number of DR clinical classes (5).
    """

    def __init__(
        self,
        root_candidates: List[Path],
        gan_root_candidates: Optional[List[Path]] = None,
        seed: int = 42,
        num_classes: int = 5,
    ) -> None:
        self.seed = seed
        self.num_classes = num_classes
        self.root = _find_candidate(root_candidates)
        if self.root is None:
            raise FileNotFoundError(
                f"APTOS root path not found among candidates: {root_candidates}"
            )
        self.gan_root = (
            _find_candidate(gan_root_candidates) if gan_root_candidates else None
        )
        self._manifest: Optional[pd.DataFrame] = None
        self._splits: dict = {}

    def load_manifest(self) -> pd.DataFrame:
        """Reads train.csv and constructs manifest DataFrame with SHA-256 hashes."""
        csv_path = self.root / "train.csv"
        df = pd.read_csv(csv_path)
        img_dir = self.root / "train_images"

        rows = []
        for _, row in df.iterrows():
            id_code = str(row["id_code"])
            label = int(row["diagnosis"])
            for ext in IMG_EXTS:
                p = img_dir / f"{id_code}{ext}"
                if p.exists():
                    rows.append(
                        {
                            "image_path": str(p),
                            "label": label,
                            "source": "real",
                            "id_code": id_code,
                            "image_hash": _file_sha256(p),
                        }
                    )
                    break
        self._manifest = pd.DataFrame(rows)
        return self._manifest

    def get_split(self, split: str) -> pd.DataFrame:
        """
        Retrieves train/val/test subset split.
        Builds splits automatically if not yet populated.
        """
        if not self._splits:
            self._build_splits()
        assert split in self._splits, f"split must be train/val/test, got {split!r}"
        return self._splits[split]

    def _build_splits(self) -> None:
        if self._manifest is None:
            self.load_manifest()

        df = self._manifest
        # Filter duplicate image hashes
        df = df.drop_duplicates(subset="image_hash").reset_index(drop=True)

        train_real, temp = train_test_split(
            df, test_size=0.40, random_state=self.seed, stratify=df["label"]
        )
        val_real, test_real = train_test_split(
            temp, test_size=0.50, random_state=self.seed, stratify=temp["label"]
        )

        # Append GAN synthetic data to training split if provided
        train_parts = [train_real]
        if self.gan_root is not None:
            gan_rows = self._load_gan(self.gan_root)
            if not gan_rows.empty:
                train_parts.append(gan_rows)

        train_df = (
            pd.concat(train_parts, ignore_index=True)
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
        )

        self._splits = {
            "train": train_df,
            "val": val_real.reset_index(drop=True),
            "test": test_real.reset_index(drop=True),
        }

    @staticmethod
    def _load_gan(gan_root: Path) -> pd.DataFrame:
        """Loads synthetic GAN images across class subfolders (0..4)."""
        rows = []
        for label in range(5):
            class_dir = gan_root / str(label)
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    rows.append(
                        {
                            "image_path": str(p),
                            "label": label,
                            "source": "gan_aptos",
                            "id_code": p.stem,
                            "image_hash": None,
                        }
                    )
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Messidor Loader
# ─────────────────────────────────────────────────────────────────────────────
class MessidorLoader(IDatasetLoader):
    """
    Loads Messidor retinal fundus dataset from CSV manifest, splitting 60/20/20.
    LSP — Interchangeable with APTOSLoader across Use Cases.
    """

    def __init__(
        self,
        root_candidates: List[Path],
        gan_root_candidates: Optional[List[Path]] = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.root = _find_candidate(root_candidates)
        if self.root is None:
            raise FileNotFoundError(
                f"Messidor root path not found among candidates: {root_candidates}"
            )
        self.gan_root = (
            _find_candidate(gan_root_candidates) if gan_root_candidates else None
        )
        self._manifest: Optional[pd.DataFrame] = None
        self._splits: dict = {}

    def load_manifest(self) -> pd.DataFrame:
        csv_path = self.root / "messidor_data.csv"
        df = pd.read_csv(csv_path)
        img_dir = self.root / "messidor-2" / "messidor-2"

        rows = []
        for _, row in df.iterrows():
            fname = str(row.get("image_id", row.get("id_code", "")))
            label = int(row.get("adjudicated_dr_grade", row.get("label", 0)))
            for ext in IMG_EXTS:
                p = img_dir / f"{fname}{ext}"
                if not p.exists():
                    p = img_dir / fname
                if p.exists():
                    rows.append(
                        {
                            "image_path": str(p),
                            "label": label,
                            "source": "real",
                            "id_code": fname,
                            "image_hash": _file_sha256(p),
                        }
                    )
                    break
        self._manifest = pd.DataFrame(rows)
        return self._manifest

    def get_split(self, split: str) -> pd.DataFrame:
        if not self._splits:
            self._build_splits()
        return self._splits[split]

    def _build_splits(self) -> None:
        if self._manifest is None:
            self.load_manifest()
        df = self._manifest.drop_duplicates(subset="image_hash").reset_index(drop=True)
        train, temp = train_test_split(
            df, test_size=0.40, random_state=self.seed, stratify=df["label"]
        )
        val, test = train_test_split(
            temp, test_size=0.50, random_state=self.seed, stratify=temp["label"]
        )
        self._splits = {
            "train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch DRDataset
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms(image_size: int, mode: str) -> dict:
    """
    Constructs RS-27 production data augmentation and preprocessing transforms.

    Returns
    -------
    dict with keys: 'real_train', 'gan_train', 'valid'
    """
    real_train = transforms.Compose(
        [
            LetterboxPad(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.20),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(
                brightness=0.08, contrast=0.08, saturation=0.06, hue=0.015
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    gan_train = transforms.Compose(
        [
            LetterboxPad(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(degrees=7, fill=0),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.04, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid = transforms.Compose(
        [
            LetterboxPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return {"real_train": real_train, "gan_train": gan_train, "valid": valid}


class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy grading.

    Source-aware augmentation:
    - Real fundus photos: processed with 'real_train' transforms.
    - GAN synthetic photos: processed with 'gan_train' transforms (lighter augmentation).
    - Validation/Test photos: processed with 'valid' transforms (deterministic letterbox pad only).

    Parameters
    ----------
    df : pd.DataFrame
        Manifest DataFrame with columns ['image_path', 'label', 'source'].
    transforms_dict : dict
        Dictionary containing transform pipelines: 'real_train', 'gan_train', 'valid'.
    mode : str
        Split mode: 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms_dict: dict,
        mode: str = "train",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tfms = transforms_dict
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def _select_transform(self, source: str) -> transforms.Compose:
        if self.mode != "train":
            return self.tfms["valid"]
        if "gan" in str(source).lower():
            return self.tfms["gan_train"]
        return self.tfms["real_train"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self._select_transform(row.get("source", "real"))(img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, label


def build_dataloader(
    df: pd.DataFrame,
    transforms_dict: dict,
    mode: str,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
) -> DataLoader:
    """Factory function creating standard DataLoader with seeded workers."""

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    ds = DRDataset(df, transforms_dict, mode=mode)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
