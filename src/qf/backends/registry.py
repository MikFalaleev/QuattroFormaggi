"""Registry of generation backends (plan C.5). Heavy adapters are registered lazily by
`cli.wiring`; light ones register themselves when `qf.backends` is imported."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import GenerationBackend

__all__ = ["BACKENDS"]

BACKENDS: Registry[type[GenerationBackend]] = Registry("backend", port=GenerationBackend)
