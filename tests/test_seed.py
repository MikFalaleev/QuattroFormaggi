from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

import numpy as np
import pytest

from qf.common import set_global_seed

seed_module = importlib.import_module("qf.common.seed")


def test_same_seed_same_sequence() -> None:
    set_global_seed(123)
    first = (random.random(), float(np.random.rand()))
    set_global_seed(123)
    assert (random.random(), float(np.random.rand())) == first


def test_torch_seeded_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake_torch = SimpleNamespace(manual_seed=calls.append)
    monkeypatch.setattr(
        seed_module, "import_optional", lambda name: fake_torch if name == "torch" else None
    )
    set_global_seed(7)
    assert calls == [7]


def test_torch_absent_is_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seed_module, "import_optional", lambda name: None)
    set_global_seed(1)
