"""Name -> implementation registries (plan C.5).

A registry maps names used in YAML configs to implementations of one port. Heavy adapters are
registered lazily as "module:attr" strings so that importing a package never imports torch.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from qf.common.errors import QFError

__all__ = ["Registry", "all_registries"]

T = TypeVar("T")

_DISCOVERABLE: list[Registry[Any]] = []


@dataclass(frozen=True)
class _LazyEntry:
    target: str
    extra: str | None


class Registry(Generic[T]):
    """Registry of implementations of one kind.

    `port` is the Protocol the implementations satisfy (None if the registry is not tied to a
    port). Discoverable registries are visible to `all_registries()` and must have unique kinds.
    """

    kind: str
    port: type[Any] | None

    def __init__(
        self, kind: str, port: type[Any] | None = None, *, discoverable: bool = True
    ) -> None:
        if discoverable and any(r.kind == kind for r in _DISCOVERABLE):
            raise QFError(f"a registry of kind '{kind}' already exists")
        self.kind = kind
        self.port = port
        self._entries: dict[str, T] = {}
        self._lazy: dict[str, _LazyEntry] = {}
        if discoverable:
            _DISCOVERABLE.append(self)

    def _check_new_name(self, name: str) -> None:
        if not name:
            raise QFError(f"{self.kind}: empty implementation name")
        if name in self._entries or name in self._lazy:
            raise QFError(f"{self.kind} '{name}' is already registered")

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: `@REGISTRY.register("name")` on a class or factory."""
        self._check_new_name(name)

        def decorator(obj: T) -> T:
            self._check_new_name(name)
            self._entries[name] = obj
            return obj

        return decorator

    def register_lazy(self, name: str, target: str, *, extra: str | None = None) -> None:
        """Register `target` ("package.module:attr"), imported only on first `get(name)`.

        `extra` names the optional-dependency group that provides the module's imports.
        """
        module, sep, attr = target.partition(":")
        if not sep or not module or not attr:
            raise QFError(
                f"{self.kind} '{name}': lazy target must be 'module:attr', got {target!r}"
            )
        self._check_new_name(name)
        self._lazy[name] = _LazyEntry(target=target, extra=extra)

    def get(self, name: str) -> T:
        if name in self._entries:
            return self._entries[name]
        if name in self._lazy:
            obj = self._resolve(name, self._lazy[name])
            self._entries[name] = obj
            return obj
        available = ", ".join(self.names()) or "none"
        raise QFError(f"unknown {self.kind} '{name}'; available: {available}")

    def _resolve(self, name: str, entry: _LazyEntry) -> T:
        module_name, _, attr = entry.target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            hint = f"; install it with `uv sync --extra {entry.extra}`" if entry.extra else ""
            raise QFError(f"{self.kind} '{name}' cannot be loaded ({exc}){hint}") from exc
        try:
            return cast(T, getattr(module, attr))
        except AttributeError as exc:
            raise QFError(f"{self.kind} '{name}': {module_name} has no attribute {attr!r}") from exc

    def names(self) -> list[str]:
        return sorted({*self._entries, *self._lazy})

    def __contains__(self, name: object) -> bool:
        return name in self._entries or name in self._lazy


def all_registries() -> list[Registry[Any]]:
    """Every discoverable registry created so far (import `qf.cli.wiring` to create them all)."""
    return list(_DISCOVERABLE)
