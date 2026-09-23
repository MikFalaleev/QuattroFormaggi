"""Artifacts exchanged between pipeline stages (plan C.6).

Every artifact (file or directory) has a sibling `<artifact>.manifest.json` holding an
`ArtifactRef`: kind, schema version, sha256, producing run and parent hashes. Paths are stored
relative to the project root so a tree copied to another machine is still readable.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from qf.common.errors import QFError
from qf.common.hashing import sha256_file, sha256_json
from qf.common.paths import (
    ARTIFACTS,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_SPLITS,
    RUNS,
    atomic_write_text,
    project_root,
)

__all__ = [
    "MANIFEST_SUFFIX",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRef",
    "artifact_sha256",
    "lineage",
    "load_artifact_manifest",
    "manifest_path_for",
    "read_artifact",
    "sha256_dir",
    "write_artifact",
]

MANIFEST_SUFFIX = ".manifest.json"
_DEFAULT_SEARCH_DIRS = (DATA_RAW, DATA_PROCESSED, DATA_SPLITS, ARTIFACTS, RUNS)

ArtifactKind = Literal[
    "raw_dataset",
    "load_facts",
    "sft_dataset",
    "benchmark",
    "predictions",
    "metrics",
    "lora_adapter",
    "hf_model",
    "gguf",
    "release",
]


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ArtifactKind
    path: PurePosixPath
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = Field(min_length=1)
    producer_run_id: str = Field(min_length=1)
    parents: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def _path_is_relative(cls, value: PurePosixPath) -> PurePosixPath:
        if value.is_absolute() or ".." in value.parts or str(value) in ("", "."):
            raise ValueError(f"artifact path must be relative to the project root: {value}")
        return value


class ArtifactManifest(BaseModel):
    """Content of `<artifact>.manifest.json`: the reference plus stage-specific details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_manifest_version: Literal["artifact_manifest_v1"] = "artifact_manifest_v1"
    ref: ArtifactRef
    extra: dict[str, Any] = Field(default_factory=dict)


def sha256_dir(path: Path) -> str:
    """Hash of the sorted list of (relative POSIX path, file sha256); independent of walk order."""
    entries = sorted(
        (file.relative_to(path).as_posix(), sha256_file(file))
        for file in path.rglob("*")
        if file.is_file()
    )
    return sha256_json(entries)


def artifact_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_dir(path)
    raise QFError(f"artifact does not exist: {path}")


def manifest_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + MANIFEST_SUFFIX)


def _resolve(path: Path, root: Path) -> tuple[Path, PurePosixPath]:
    """Return (absolute path, path relative to root); reject paths outside the root."""
    absolute = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = absolute.relative_to(root.resolve())
    except ValueError as exc:
        raise QFError(f"artifact {path} is outside the project root {root}") from exc
    return absolute, PurePosixPath(relative.as_posix())


def write_artifact(
    path: Path,
    kind: ArtifactKind,
    schema_version: str,
    run_id: str,
    parents: Iterable[str] = (),
    *,
    extra: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> ArtifactRef:
    """Hash an existing file/directory and write its manifest next to it."""
    root = root if root is not None else project_root()
    absolute, relative = _resolve(path, root)
    try:
        ref = ArtifactRef(
            kind=kind,
            path=relative,
            sha256=artifact_sha256(absolute),
            schema_version=schema_version,
            producer_run_id=run_id,
            parents=tuple(parents),
        )
    except ValidationError as exc:
        raise QFError(f"invalid artifact reference for {path}: {exc}") from exc
    manifest = ArtifactManifest(ref=ref, extra=dict(extra or {}))
    atomic_write_text(manifest_path_for(absolute), manifest.model_dump_json(indent=2) + "\n")
    return ref


def load_artifact_manifest(path: Path, *, root: Path | None = None) -> ArtifactManifest:
    root = root if root is not None else project_root()
    absolute, _ = _resolve(path, root)
    manifest_path = manifest_path_for(absolute)
    try:
        return ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QFError(f"missing or unreadable artifact manifest {manifest_path}: {exc}") from exc
    except ValidationError as exc:
        raise QFError(f"invalid artifact manifest {manifest_path}: {exc}") from exc


def read_artifact(
    path: Path,
    expected_kind: ArtifactKind,
    supported_versions: Collection[str],
    *,
    root: Path | None = None,
) -> ArtifactRef:
    """Load and verify an artifact: kind, schema version, location and content hash."""
    root = root if root is not None else project_root()
    absolute, relative = _resolve(path, root)
    ref = load_artifact_manifest(absolute, root=root).ref
    if ref.kind != expected_kind:
        raise QFError(f"{relative}: expected artifact kind '{expected_kind}', found '{ref.kind}'")
    if ref.schema_version not in supported_versions:
        supported = ", ".join(sorted(supported_versions)) or "none"
        raise QFError(
            f"{relative}: unsupported schema version '{ref.schema_version}' "
            f"for '{ref.kind}' (supported: {supported})"
        )
    if ref.path != relative:
        raise QFError(f"{relative}: manifest describes a different location ({ref.path})")
    actual = artifact_sha256(absolute)
    if actual != ref.sha256:
        raise QFError(f"{relative}: content hash mismatch (manifest {ref.sha256}, actual {actual})")
    return ref


def _index_manifests(search_roots: Sequence[Path], root: Path) -> dict[str, ArtifactRef]:
    index: dict[str, ArtifactRef] = {}
    for search_root in search_roots:
        base = search_root if search_root.is_absolute() else root / search_root
        if not base.is_dir():
            continue
        for manifest_path in base.rglob(f"*{MANIFEST_SUFFIX}"):
            try:
                text = manifest_path.read_text(encoding="utf-8")
                manifest = ArtifactManifest.model_validate_json(text)
            except (OSError, ValidationError) as exc:
                raise QFError(f"invalid artifact manifest {manifest_path}: {exc}") from exc
            index[manifest.ref.sha256] = manifest.ref
    return index


def lineage(
    ref: ArtifactRef,
    *,
    search_roots: Sequence[Path] | None = None,
    root: Path | None = None,
) -> list[ArtifactRef]:
    """All ancestors of `ref` (breadth-first, parents first), found via manifests on disk."""
    root = root if root is not None else project_root()
    index = _index_manifests(search_roots or _DEFAULT_SEARCH_DIRS, root)
    ancestors: list[ArtifactRef] = []
    seen: set[str] = {ref.sha256}
    queue: deque[str] = deque(ref.parents)
    while queue:
        parent_hash = queue.popleft()
        if parent_hash in seen:
            continue
        seen.add(parent_hash)
        parent = index.get(parent_hash)
        if parent is None:
            raise QFError(f"lineage of {ref.path}: parent artifact {parent_hash} not found")
        ancestors.append(parent)
        queue.extend(parent.parents)
    return ancestors
