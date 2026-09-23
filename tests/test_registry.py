from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from qf.common import QFError, Registry, all_registries

registry_module = importlib.import_module("qf.common.registry")


def _registry() -> Registry[type]:
    return Registry("test_kind", discoverable=False)


def test_register_and_get() -> None:
    registry = _registry()

    @registry.register("alpha")
    class Alpha:
        pass

    assert registry.get("alpha") is Alpha
    assert "alpha" in registry
    assert "beta" not in registry
    assert registry.names() == ["alpha"]


def test_duplicate_name_raises() -> None:
    registry = _registry()
    registry.register("alpha")(int)
    with pytest.raises(QFError, match="already registered"):
        registry.register("alpha")
    with pytest.raises(QFError, match="already registered"):
        registry.register_lazy("alpha", "json:loads")
    with pytest.raises(QFError, match="empty"):
        registry.register("")


def test_unknown_name_lists_available() -> None:
    registry = _registry()
    registry.register("alpha")(int)
    registry.register_lazy("beta", "json:loads")
    with pytest.raises(QFError) as info:
        registry.get("gamma")
    assert "unknown test_kind 'gamma'" in str(info.value)
    assert "alpha, beta" in str(info.value)


def test_register_lazy_imports_only_on_get(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "qf_lazy_fixture_module"
    (tmp_path / f"{module_name}.py").write_text("class Impl:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    registry = _registry()
    registry.register_lazy("impl", f"{module_name}:Impl", extra="train-cuda")
    assert registry.names() == ["impl"]
    assert module_name not in sys.modules
    impl = registry.get("impl")
    assert module_name in sys.modules
    assert impl.__name__ == "Impl"
    assert registry.get("impl") is impl
    monkeypatch.delitem(sys.modules, module_name)


def test_lazy_missing_module_names_the_extra() -> None:
    registry = _registry()
    registry.register_lazy("heavy", "qf_no_such_module_xyz:Impl", extra="train-cuda")
    with pytest.raises(QFError, match="uv sync --extra train-cuda"):
        registry.get("heavy")


def test_lazy_missing_attribute() -> None:
    registry = _registry()
    registry.register_lazy("broken", "json:no_such_attribute")
    with pytest.raises(QFError, match="no attribute"):
        registry.get("broken")


@pytest.mark.parametrize("target", ["json", "json:", ":loads", "json.loads"])
def test_lazy_target_format(target: str) -> None:
    with pytest.raises(QFError, match="module:attr"):
        _registry().register_lazy("x", target)


def test_discoverable_kinds_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "_DISCOVERABLE", [])
    first = Registry("unique_kind")
    assert all_registries() == [first]
    with pytest.raises(QFError, match="already exists"):
        Registry("unique_kind")
    Registry("unique_kind", discoverable=False)  # private registries may reuse a kind
    assert all_registries() == [first]
