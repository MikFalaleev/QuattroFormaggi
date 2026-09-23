"""Optional-dependency helpers.

`qf.common` must not import heavy libraries (torch, transformers, ...) statically: the
import-linter contract forbids it (IMPLEMENTATION_PLAN.md C.3). Code that needs to *probe*
such a library (doctor, seeding) goes through these helpers, which import by name at runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = ["import_optional", "is_installed"]


def is_installed(name: str) -> bool:
    """Return True if a top-level module can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def import_optional(name: str) -> ModuleType | None:
    """Import a top-level module if it is installed, else return None."""
    if not is_installed(name):
        return None
    return importlib.import_module(name)
