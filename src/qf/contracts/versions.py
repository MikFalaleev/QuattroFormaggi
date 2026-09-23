"""Schema versions the current code can read, per artifact kind (plan C.7).

This is the only place listing supported versions. Each step that introduces an artifact
format adds its entry; an unknown version is an explicit error, never a guess.
"""

from __future__ import annotations

from qf.common import QFError

__all__ = ["SUPPORTED_SCHEMA_VERSIONS", "require_supported", "supported_versions"]

SUPPORTED_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {}


def supported_versions(kind: str) -> frozenset[str]:
    versions = SUPPORTED_SCHEMA_VERSIONS.get(kind)
    if not versions:
        raise QFError(f"no supported schema versions are registered for artifact kind '{kind}'")
    return versions


def require_supported(kind: str, version: str) -> None:
    versions = supported_versions(kind)
    if version not in versions:
        listed = ", ".join(sorted(versions))
        raise QFError(f"unsupported schema version '{version}' for '{kind}' (supported: {listed})")
