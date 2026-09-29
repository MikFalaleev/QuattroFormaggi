"""Registries of export tools (plan steps 16–17): mergers, GGUF converters, quantizers.
Implementations register lazily in `cli.wiring`."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import Merger, ModelConverter, Quantizer

__all__ = ["CONVERTERS", "MERGERS", "QUANTIZERS"]

MERGERS: Registry[type[Merger]] = Registry("merger", port=Merger)
CONVERTERS: Registry[type[ModelConverter]] = Registry("converter", port=ModelConverter)
QUANTIZERS: Registry[type[Quantizer]] = Registry("quantizer", port=Quantizer)
