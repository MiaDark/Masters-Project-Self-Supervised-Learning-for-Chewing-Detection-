"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs.

    Call this once at the start of every training/eval entry point and log the
    seed to MLflow so any run can be reproduced exactly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed yet — fine for the data-only stage.
        pass
