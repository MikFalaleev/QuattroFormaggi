"""Per-run reproducibility manifest (`run_manifest.json`), plan section 8 / rule A.2.7."""

from __future__ import annotations

import importlib.metadata
import re
import secrets
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qf.common.errors import QFError
from qf.common.paths import RUNS, atomic_write_text

__all__ = [
    "MANIFEST_FILENAME",
    "RunManifest",
    "RunRecorder",
    "RunStatus",
    "collect_git_info",
    "collect_package_versions",
    "new_run_id",
    "read_manifest",
    "start_run",
    "update_manifest",
    "write_manifest",
]

MANIFEST_FILENAME = "run_manifest.json"
_KIND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_GIT_TIMEOUT_S = 10

RunStatus = Literal["running", "completed", "failed", "aborted"]


class RunManifest(BaseModel):
    """Everything needed to reproduce or audit one run."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal["run_manifest_v1"] = "run_manifest_v1"
    run_id: str
    kind: str
    created_at: datetime
    status: RunStatus = "running"
    experiment_id: str | None = None  # a registered experiment (training runs, D-118)
    git_commit: str | None = None
    git_dirty: bool = False
    hardware: dict[str, Any] = Field(default_factory=dict)
    package_versions: dict[str, str | None] = Field(default_factory=dict)
    base_revision: str | None = None
    data_hashes: dict[str, str] = Field(default_factory=dict)
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    parent_checkpoint: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    wall_time_s: float | None = None
    peak_memory_bytes: int | None = None


def new_run_id(kind: str) -> str:
    """Return `YYYYMMDD-HHMMSS-<kind>-<6 hex>` (UTC time)."""
    if not _KIND_PATTERN.match(kind):
        raise QFError(f"invalid run kind {kind!r}: use lowercase letters, digits, '-' and '_'")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{kind}-{secrets.token_hex(3)}"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_git_info(cwd: Path | None = None) -> tuple[str | None, bool]:
    """Return (HEAD commit or None, working tree dirty). Never raises outside a repository."""
    where = cwd if cwd is not None else Path.cwd()
    if _git(["rev-parse", "--is-inside-work-tree"], where) != "true":
        return None, False
    commit = _git(["rev-parse", "HEAD"], where)
    status = _git(["status", "--porcelain"], where)
    return commit, bool(status)


def collect_package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    path = run_dir / MANIFEST_FILENAME
    atomic_write_text(path, manifest.model_dump_json(indent=2) + "\n")
    return path


def read_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / MANIFEST_FILENAME
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QFError(f"cannot read {path}: {exc}") from exc
    except ValidationError as exc:
        raise QFError(f"invalid run manifest {path}: {exc}") from exc


def update_manifest(run_dir: Path, **fields: Any) -> RunManifest:
    """Merge `fields` into the stored manifest, validate and write it back atomically."""
    current = read_manifest(run_dir)
    if "run_id" in fields and fields["run_id"] != current.run_id:
        raise QFError(f"run_id of {run_dir} cannot be changed")
    data = current.model_dump()
    data.update(fields)
    try:
        updated = RunManifest.model_validate(data)
    except ValidationError as exc:
        raise QFError(f"invalid manifest update for {run_dir}: {exc}") from exc
    write_manifest(updated, run_dir)
    return updated


_BASE_PACKAGES = ("quattro-formaggi",)


@dataclass(frozen=True)
class RunRecorder:
    """Common bookkeeping of one run: id, start time and the final `run_manifest.json`."""

    kind: str
    run_id: str
    root: Path
    created_at: datetime
    started_monotonic: float

    @property
    def run_dir(self) -> Path:
        return self.root / RUNS / self.run_id

    def finish(
        self, *, packages: Iterable[str] = (), status: RunStatus = "completed", **fields: Any
    ) -> RunManifest:
        """Write the manifest with git state, package versions and wall time filled in."""
        git_commit, git_dirty = collect_git_info(self.root)
        try:
            manifest = RunManifest(
                run_id=self.run_id,
                kind=self.kind,
                created_at=self.created_at,
                status=status,
                git_commit=git_commit,
                git_dirty=git_dirty,
                package_versions=collect_package_versions([*_BASE_PACKAGES, *packages]),
                wall_time_s=round(time.monotonic() - self.started_monotonic, 3),
                **fields,
            )
        except ValidationError as exc:
            raise QFError(f"invalid manifest for run {self.run_id}: {exc}") from exc
        write_manifest(manifest, self.run_dir)
        return manifest


def start_run(kind: str, root: Path) -> RunRecorder:
    """Begin a run under `root/runs/`; call `finish()` once the run has produced its outputs."""
    return RunRecorder(
        kind=kind,
        run_id=new_run_id(kind),
        root=root,
        created_at=datetime.now(UTC),
        started_monotonic=time.monotonic(),
    )
