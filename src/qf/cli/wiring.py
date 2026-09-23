"""Composition root: the only place where implementations are chosen (plan C.3, C.5).

Later steps register heavy adapters here lazily, e.g.
`BACKENDS.register_lazy("hf_local", "qf.backends.hf_local:HFLocalBackend", extra="train-cuda")`.
Adapter modules are referenced only as strings: this module never imports them at top level.
Importing this module makes every registry (and every lazy entry) visible to
`qf.common.all_registries()`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

import qf.data  # noqa: F401  (imported for its registries: RAW_SOURCES)
import qf.domain  # noqa: F401  (imported for its registries: TASKS)
from qf.common import ComponentConfig, QFError, Registry

__all__ = ["build"]


def build(registry: Registry[Any], selector: ComponentConfig) -> Any:
    """Instantiate `selector.name` from `registry` with its own validated `Config`."""
    implementation = registry.get(selector.name)
    config_model = getattr(implementation, "Config", None)
    if not (isinstance(config_model, type) and issubclass(config_model, BaseModel)):
        raise QFError(f"{registry.kind} '{selector.name}' has no nested pydantic `Config` model")
    try:
        params = config_model.model_validate(selector.model_extra or {})
    except ValidationError as exc:
        raise QFError(f"{registry.kind} '{selector.name}': invalid parameters: {exc}") from exc
    return implementation(params)
