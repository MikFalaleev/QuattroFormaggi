from __future__ import annotations

import importlib
import json

import pytest

optional = importlib.import_module("qf.common._optional")


def test_is_installed() -> None:
    assert optional.is_installed("json")
    assert not optional.is_installed("qf_definitely_missing_module")


def test_is_installed_tolerates_broken_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> None:
        raise ValueError("__spec__ is None")

    monkeypatch.setattr(optional.importlib.util, "find_spec", broken)
    assert not optional.is_installed("anything")


def test_import_optional() -> None:
    assert optional.import_optional("json") is json
    assert optional.import_optional("qf_definitely_missing_module") is None
