"""Registries of the data stages (plan C.5). Implementations register themselves by name."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import RawSource

__all__ = ["RAW_SOURCES"]

RAW_SOURCES: Registry[type[RawSource]] = Registry("raw_source", port=RawSource)
