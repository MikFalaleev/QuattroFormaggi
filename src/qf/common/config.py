"""YAML configuration loading into strict Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from qf.common.errors import QFError
from qf.common.hashing import sha256_json

__all__ = ["ComponentConfig", "StrictConfig", "config_hash", "load_yaml_config"]


class StrictConfig(BaseModel):
    """Base for every config model: unknown keys are errors, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentConfig(BaseModel):
    """Open `{name, **params}` selector for a registered implementation (plan C.5).

    Extra keys are the implementation's own parameters; the implementation validates them
    with its nested `Config` model, so adding an implementation never edits this class.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    )


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load YAML at `path` into `model`, which must forbid unknown keys."""
    if model.model_config.get("extra") != "forbid":
        raise QFError(f"config model {model.__name__} must set extra='forbid'")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QFError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise QFError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise QFError(f"{path}: top level of a config must be a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise QFError(f"{path}: {_format_validation_error(exc)}") from exc


def config_hash(cfg: BaseModel) -> str:
    return sha256_json(cfg.model_dump(mode="json"))
