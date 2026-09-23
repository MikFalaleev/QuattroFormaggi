"""Registries of the data stages (plan C.5). Implementations register themselves by name."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import FactsBuilder, RawSource

__all__ = ["FACTS_BUILDERS", "RAW_SOURCES"]

RAW_SOURCES: Registry[type[RawSource]] = Registry("raw_source", port=RawSource)
FACTS_BUILDERS: Registry[type[FactsBuilder]] = Registry("facts_builder", port=FactsBuilder)
