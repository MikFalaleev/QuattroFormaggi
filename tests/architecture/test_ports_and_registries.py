from __future__ import annotations

import importlib
import types
from typing import Protocol

from qf.common import Registry, all_registries
from tests.architecture.helpers import (
    discover_ports,
    implementations_table,
    ports_without_implementation,
    undocumented_names,
)
from tests.conftest import REPO_ROOT


def _load_everything() -> None:
    importlib.import_module("qf.cli.wiring")  # makes every registry and lazy entry visible


def test_every_port_has_at_least_one_impl() -> None:
    _load_everything()
    ports = discover_ports(importlib.import_module("qf.contracts.ports"))
    assert ports_without_implementation(ports, all_registries()) == []


def test_registry_names_unique_and_documented() -> None:
    _load_everything()
    registries = all_registries()
    kinds = [r.kind for r in registries]
    assert len(kinds) == len(set(kinds))
    doc = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    table = implementations_table(doc)
    assert table, "docs/ARCHITECTURE.md has no implementations table"
    assert undocumented_names(registries, table) == []


class _DummyPort(Protocol):
    def run(self) -> int: ...


def test_port_checker_detects_missing_implementation() -> None:
    registry: Registry[type] = Registry("dummy", port=_DummyPort, discoverable=False)
    assert ports_without_implementation([_DummyPort], [registry]) == ["_DummyPort"]
    registry.register("impl")(int)
    assert ports_without_implementation([_DummyPort], [registry]) == []


def test_discover_ports_finds_only_local_protocols() -> None:
    module = types.ModuleType("fake_ports")
    exec(
        "from typing import Protocol\n"
        "class LocalPort(Protocol):\n    def f(self) -> int: ...\n"
        "class NotAPort:\n    pass\n",
        module.__dict__,
    )
    assert [p.__name__ for p in discover_ports(module)] == ["LocalPort"]


def test_documentation_checker_detects_missing_name() -> None:
    registry: Registry[type] = Registry("dummy", discoverable=False)
    registry.register("zzz_undocumented")(int)
    doc = "# A\n## Таблица реализаций\n| port | `other` |\n## Next\n`zzz_undocumented`\n"
    table = implementations_table(doc)
    assert "`zzz_undocumented`" not in table
    assert undocumented_names([registry], table) == ["dummy:zzz_undocumented"]
