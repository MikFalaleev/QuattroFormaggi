"""Registry of mergers (plan step 16). Heavy implementations register lazily in `cli.wiring`."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import Merger

__all__ = ["MERGERS"]

MERGERS: Registry[type[Merger]] = Registry("merger", port=Merger)
