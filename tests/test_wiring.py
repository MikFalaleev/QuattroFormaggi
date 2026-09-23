from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from qf.cli.wiring import build
from qf.common import ComponentConfig, QFError, Registry


class SizedImpl:
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        size: int

    def __init__(self, config: SizedImpl.Config) -> None:
        self.config = config


class NoConfigImpl:
    pass


def _registry() -> Registry[Any]:
    registry: Registry[Any] = Registry("wiring_test", discoverable=False)
    registry.register("sized")(SizedImpl)
    registry.register("no_config")(NoConfigImpl)
    return registry


def test_build_validates_params_with_implementation_config() -> None:
    built = build(_registry(), ComponentConfig(name="sized", size=3))
    assert isinstance(built, SizedImpl)
    assert built.config.size == 3


def test_build_rejects_bad_params() -> None:
    with pytest.raises(QFError, match="wiring_test 'sized': invalid parameters"):
        build(_registry(), ComponentConfig(name="sized", size="big", colour="red"))


def test_build_requires_config_model() -> None:
    with pytest.raises(QFError, match="no nested pydantic `Config`"):
        build(_registry(), ComponentConfig(name="no_config"))


def test_build_unknown_name() -> None:
    with pytest.raises(QFError, match="unknown wiring_test 'missing'"):
        build(_registry(), ComponentConfig(name="missing"))
