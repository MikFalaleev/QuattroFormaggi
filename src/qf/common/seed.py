"""Global seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np

from qf.common._optional import import_optional

__all__ = ["set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Seed python `random`, numpy and, if installed, torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    # torch is imported by name at runtime: qf.common must not import it statically (plan C.3).
    torch = import_optional("torch")
    if torch is not None:
        torch.manual_seed(seed)
