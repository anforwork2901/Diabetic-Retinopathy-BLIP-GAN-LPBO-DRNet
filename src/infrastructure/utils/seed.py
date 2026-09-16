"""
Utility: Seed Everything
SRP — Ensures deterministic experiment reproducibility across all dependencies.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Fixes random seed across Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int
        Integer seed value. Default: 42 (RS-27 verified).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Enable deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
