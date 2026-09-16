"""
Domain Entity: FundusImage
SRP — Represents a fundus photograph entity with associated label and metadata.
      Completely decoupled from any ML/DL framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .dr_grade import DRGrade


@dataclass(frozen=True)
class FundusImage:
    """
    Immutable entity / value object representing a retinal fundus photograph.

    Attributes
    ----------
    image_path : Path
        Absolute filesystem path to the image file.
    grade : DRGrade
        Ground-truth Diabetic Retinopathy clinical grade.
    source : str
        Origin of the image: 'real', 'gan_aptos', 'gan_messidor', etc.
    id_code : str
        Unique identifier code for the image (e.g., filename without extension).
    image_hash : Optional[str]
        SHA-256 hash for de-duplication and data leakage prevention.
    """

    image_path: Path
    grade: DRGrade
    source: str = "real"
    id_code: str = ""
    image_hash: Optional[str] = None

    def __post_init__(self) -> None:
        # Ensure image_path is always a Path object
        object.__setattr__(self, "image_path", Path(self.image_path))

    @property
    def label(self) -> int:
        """Integer class label corresponding to the DRGrade."""
        return int(self.grade)

    @property
    def is_synthetic(self) -> bool:
        """True if the image was synthesized via GAN augmentation."""
        return "gan" in self.source.lower()

    @property
    def exists(self) -> bool:
        """Checks whether the image file exists on disk."""
        return self.image_path.exists()
