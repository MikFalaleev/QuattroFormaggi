"""Static checks behind the architecture tests (IMPLEMENTATION_PLAN.md C.3, C.8)."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from qf.common import Registry
from tests.conftest import REPO_ROOT

SRC = REPO_ROOT / "src" / "qf"
PACKAGES = (
    "common", "contracts", "domain", "backends", "data", "eval", "training", "export",
    "runtime", "lab", "cli",
)  # fmt: skip
# The composition root may import adapter modules by path (plan C.3).
PUBLIC_API_EXEMPT = (Path("cli") / "wiring.py",)
IMPLEMENTATIONS_HEADING = "## Таблица реализаций"


@dataclass(frozen=True)
class ImportRef:
    module: str  # absolute module name
    names: tuple[str, ...]  # names in `from module import ...`; empty for `import module`
    lineno: int


def python_files(package: str, src: Path = SRC) -> list[Path]:
    return sorted((src / package).rglob("*.py"))


def module_name(path: Path, src: Path = SRC) -> str:
    parts = list(path.relative_to(src.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path: Path, src: Path = SRC) -> list[ImportRef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = module_name(path, src).split(".")
    if path.name != "__init__.py":
        package_parts = package_parts[:-1]
    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs += [ImportRef(alias.name, (), node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            refs.append(ImportRef(module, tuple(a.name for a in node.names), node.lineno))
    return refs


def _is_submodule(package: str, name: str, src: Path) -> bool:
    # Compare exact names: macOS file systems are case-insensitive, so a Path.exists() check
    # would mistake the class `Registry` for the module `registry.py`.
    entries = {entry.name for entry in (src / package).iterdir()}
    return f"{name}.py" in entries or (name in entries and (src / package / name).is_dir())


def public_api_violations(package: str, src: Path = SRC) -> list[str]:
    """Imports from another qf package that bypass its root `__init__` public API."""
    violations = []
    exempt = {src / relative for relative in PUBLIC_API_EXEMPT}
    for path in python_files(package, src):
        if path in exempt:
            continue
        for ref in imports_of(path, src):
            parts = ref.module.split(".")
            if parts[0] != "qf" or len(parts) < 2 or parts[1] == package:
                continue
            deep = len(parts) > 2
            via_root = [n for n in ref.names if len(parts) == 2 and _is_submodule(parts[1], n, src)]
            if deep or via_root:
                target = ref.module + (f" ({', '.join(via_root)})" if via_root else "")
                violations.append(f"{path.relative_to(src.parent)}:{ref.lineno} imports {target}")
    return violations


def lab_isolation_violations() -> list[str]:
    violations = []
    for package in PACKAGES:
        for path in python_files(package):
            for ref in imports_of(path):
                parts = ref.module.split(".")
                if parts[0] != "qf" or len(parts) < 2:
                    continue
                if package == "lab" and parts[1] not in ("lab", "common"):
                    violations.append(f"{path.relative_to(REPO_ROOT)} imports {ref.module}")
                if package != "lab" and parts[1] == "lab":
                    violations.append(f"{path.relative_to(REPO_ROOT)} imports {ref.module}")
    return violations


def discover_ports(module: ModuleType) -> list[type[Any]]:
    """Protocol classes defined (not merely imported) in `module`."""
    return [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and getattr(obj, "_is_protocol", False)
        and obj.__module__ == module.__name__
    ]


def ports_without_implementation(
    ports: Iterable[type[Any]], registries: Iterable[Registry[Any]]
) -> list[str]:
    """A port extension (a Protocol based on another port, e.g. `BatchGenerationBackend`) counts
    as implemented through the registry of the port it extends."""
    registries = list(registries)
    return [
        port.__name__
        for port in ports
        if not any(r.port in port.__mro__ and r.names() for r in registries)
    ]


def implementations_table(architecture_doc: str) -> str:
    """Text of the implementations section of docs/ARCHITECTURE.md."""
    start = architecture_doc.find(IMPLEMENTATIONS_HEADING)
    if start < 0:
        return ""
    end = architecture_doc.find("\n## ", start + len(IMPLEMENTATIONS_HEADING))
    return architecture_doc[start : end if end >= 0 else None]


def undocumented_names(registries: Iterable[Registry[Any]], table: str) -> list[str]:
    return [
        f"{r.kind}:{name}" for r in registries for name in r.names() if f"`{name}`" not in table
    ]
