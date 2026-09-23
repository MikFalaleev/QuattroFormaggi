"""Registries of the data stages (plan C.5). Implementations register themselves by name."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import FactsBuilder, HardCase, RawSource, TemplateFamily

__all__ = ["FACTS_BUILDERS", "HARD_CASES", "RAW_SOURCES", "TEMPLATE_FAMILIES"]

RAW_SOURCES: Registry[type[RawSource]] = Registry("raw_source", port=RawSource)
FACTS_BUILDERS: Registry[type[FactsBuilder]] = Registry("facts_builder", port=FactsBuilder)
TEMPLATE_FAMILIES: Registry[type[TemplateFamily]] = Registry("template_family", port=TemplateFamily)
HARD_CASES: Registry[type[HardCase]] = Registry("hard_case", port=HardCase)
