"""
Domain Entity: DRGrade
SRP — Defines the core domain representation of Diabetic Retinopathy severity grades.
      Completely decoupled from any ML/DL framework.
"""
from __future__ import annotations

from enum import IntEnum


class DRGrade(IntEnum):
    """Diabetic Retinopathy clinical severity grade (0 to 4)."""

    NO_DR = 0           # No apparent retinopathy
    MILD = 1            # Mild non-proliferative DR
    MODERATE = 2        # Moderate non-proliferative DR
    SEVERE = 3          # Severe non-proliferative DR
    PROLIFERATIVE = 4   # Proliferative DR

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────
    @property
    def display_name(self) -> str:
        """Full human-readable name for each grade."""
        return {
            DRGrade.NO_DR: "No DR",
            DRGrade.MILD: "Mild",
            DRGrade.MODERATE: "Moderate",
            DRGrade.SEVERE: "Severe",
            DRGrade.PROLIFERATIVE: "Proliferative DR",
        }[self]

    @property
    def is_referable(self) -> bool:
        """Returns True if the grade requires specialist referral (>= Moderate)."""
        return self >= DRGrade.MODERATE

    @classmethod
    def all_names(cls) -> list[str]:
        """List of all grade display names in ascending order."""
        return [g.display_name for g in cls]

    @classmethod
    def num_classes(cls) -> int:
        """Total number of classification classes."""
        return len(cls)
