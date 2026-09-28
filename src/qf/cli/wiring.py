"""Composition root: the only place where implementations are chosen (plan C.3, C.5).

Heavy adapters are registered here lazily (the trainer of step 12, the HF backend of step 14,
the merger of step 16).
Adapter modules are referenced only as strings: this module never imports them at top level.
Importing this module makes every registry (and every lazy entry) visible to
`qf.common.all_registries()`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

import qf.backends  # noqa: F401  (imported for its registries: BACKENDS)
import qf.baselines  # noqa: F401  (registers the lower-bound baselines in BACKENDS, D-109)
import qf.data  # noqa: F401  (imported for its registries: RAW_SOURCES)
import qf.domain  # noqa: F401  (imported for its registries: TASKS)
import qf.eval  # noqa: F401  (imported for its registries: METRICS)
from qf.backends import BACKENDS
from qf.common import ComponentConfig, QFError, Registry
from qf.export import MERGERS
from qf.training import TRAINERS

__all__ = ["build"]

TRAINERS.register_lazy("hf_trainer_qlora", "qf.training.trainers.hf_qlora:HFQLoRATrainer",
                       extra="train-cuda")  # fmt: skip
BACKENDS.register_lazy("hf_local", "qf.backends.hf_local:HFLocalBackend", extra="train-cuda")
MERGERS.register_lazy("peft_merge", "qf.export.mergers.peft_merge:PeftMerger", extra="train-cuda")


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
