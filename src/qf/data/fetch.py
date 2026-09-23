"""Stage `fetch`: download the pinned raw dataset and record its provenance (plan step 2).

Only the files listed in the source config are downloaded. Files are fetched into a staging
directory next to the target and moved into place in one rename, so an interrupted download
never leaves a half-filled dataset directory (and the downloader's own cache metadata never
becomes part of the artifact).
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from qf.common import (
    DATA_RAW,
    IGNORED_FILE_NAMES,
    ArtifactRef,
    ComponentConfig,
    QFError,
    StrictConfig,
    artifact_sha256,
    atomic_write_text,
    load_artifact_manifest,
    project_root,
    sha256_file,
    start_run,
    write_artifact,
)
from qf.contracts import RawSource
from qf.data.registries import RAW_SOURCES

__all__ = [
    "PROVENANCE_FILENAME",
    "RAW_SCHEMA_VERSION",
    "FetchResult",
    "FileRecord",
    "HFDatasetSource",
    "Provenance",
    "SourceConfig",
    "SourceFileConfig",
    "fetch_raw_dataset",
    "raw_dataset_path",
    "read_provenance",
    "verify_provenance",
    "verify_raw_dataset",
    "write_provenance",
]

PROVENANCE_FILENAME = "provenance.json"
RAW_SCHEMA_VERSION = "logistics_ops_csv_v1"
_STAGING_PREFIX = ".fetch-staging-"
_RUN_KIND = "fetch"
_FORBIDDEN_NAME_PARTS = ("/", "\\", "..")


class SourceConfig(StrictConfig):
    """Parameters of the `hf_dataset` source (the `source:` block of configs/data/source.yaml)."""

    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    repo_type: Literal["dataset"] = "dataset"
    revision: str
    license: str = Field(min_length=1)
    files: tuple[str, ...] = Field(min_length=1)
    excluded_reason: dict[str, str] = Field(default_factory=dict)

    @field_validator("revision")
    @classmethod
    def _full_commit_sha(cls, value: str) -> str:
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(
                "revision must be a full 40-character lowercase commit sha, "
                f"not a branch, tag or short sha: {value!r}"
            )
        return value

    @field_validator("files")
    @classmethod
    def _plain_unique_names(cls, files: tuple[str, ...]) -> tuple[str, ...]:
        for name in files:
            if not name or any(part in name for part in _FORBIDDEN_NAME_PARTS):
                raise ValueError(f"file names must be plain top-level names: {name!r}")
            if name in IGNORED_FILE_NAMES:
                raise ValueError(f"{name!r} cannot be part of a dataset")
        if len(set(files)) != len(files):
            raise ValueError("file names must be unique")
        return files

    @model_validator(mode="after")
    def _excluded_files_not_fetched(self) -> SourceConfig:
        both = sorted(set(self.files) & set(self.excluded_reason))
        if both:
            raise ValueError(f"files are both fetched and excluded: {', '.join(both)}")
        return self


class SourceFileConfig(StrictConfig):
    """Top level of configs/data/source.yaml: selects a RawSource implementation by name."""

    source: ComponentConfig


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class Provenance(BaseModel):
    """Content of provenance.json next to the downloaded files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_version: Literal["provenance_v1"] = "provenance_v1"
    repo_id: str
    repo_type: str
    revision: str
    license: str
    fetched_at: datetime
    files: dict[str, FileRecord]
    excluded: dict[str, str]


def write_provenance(dest: Path, cfg: SourceConfig) -> Path:
    """Hash every configured file in `dest` and write provenance.json there."""
    records: dict[str, FileRecord] = {}
    for name in cfg.files:
        path = dest / name
        if not path.is_file():
            raise QFError(f"cannot write provenance: {path} is missing")
        records[name] = FileRecord(sha256=sha256_file(path), bytes=path.stat().st_size)
    provenance = Provenance(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        revision=cfg.revision,
        license=cfg.license,
        fetched_at=datetime.now(UTC),
        files=records,
        excluded=dict(cfg.excluded_reason),
    )
    path = dest / PROVENANCE_FILENAME
    atomic_write_text(path, provenance.model_dump_json(indent=2) + "\n")
    return path


def read_provenance(dest: Path) -> Provenance:
    path = dest / PROVENANCE_FILENAME
    try:
        return Provenance.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QFError(f"{PROVENANCE_FILENAME} is missing or unreadable in {dest}: {exc}") from exc
    except ValidationError as exc:
        raise QFError(f"invalid {PROVENANCE_FILENAME} in {dest}: {exc}") from exc


def verify_provenance(dest: Path) -> list[str]:
    """Recompute sizes and hashes; report missing, changed and unexpected files."""
    try:
        provenance = read_provenance(dest)
    except QFError as exc:
        return [str(exc)]
    problems: list[str] = []
    for name, record in provenance.files.items():
        path = dest / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        size = path.stat().st_size
        if size != record.bytes:
            problems.append(f"{name}: size {size} bytes, provenance says {record.bytes}")
        elif sha256_file(path) != record.sha256:
            problems.append(f"{name}: sha256 differs from provenance")
    expected = {*provenance.files, PROVENANCE_FILENAME, *IGNORED_FILE_NAMES}
    problems += [
        f"{entry.name}: unexpected file" for entry in sorted(dest.iterdir())
        if entry.name not in expected
    ]  # fmt: skip
    return problems


def _config_problems(provenance: Provenance, cfg: SourceConfig) -> list[str]:
    problems = []
    for field in ("repo_id", "revision", "license"):
        stored, configured = getattr(provenance, field), getattr(cfg, field)
        if stored != configured:
            problems.append(f"{field}: provenance has {stored!r}, config has {configured!r}")
    if set(provenance.files) != set(cfg.files):
        problems.append(
            f"files: provenance has {sorted(provenance.files)}, config has {sorted(cfg.files)}"
        )
    return problems


class Downloader(Protocol):
    def __call__(
        self, repo_id: str, filename: str, *, repo_type: str, revision: str, local_dir: Path
    ) -> object: ...


def _hf_download(
    repo_id: str, filename: str, *, repo_type: str, revision: str, local_dir: Path
) -> object:
    return hf_hub_download(
        repo_id, filename, repo_type=repo_type, revision=revision, local_dir=local_dir
    )


@RAW_SOURCES.register("hf_dataset")
class HFDatasetSource:
    """RawSource for a dataset on the Hugging Face Hub, pinned to a full commit sha."""

    Config = SourceConfig

    def __init__(
        self,
        config: SourceConfig,
        *,
        root: Path | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self.config = config
        self._root = root
        self._downloader: Downloader = downloader if downloader is not None else _hf_download

    def _project_root(self) -> Path:
        return self._root if self._root is not None else project_root()

    def target(self, dest_root: Path) -> Path:
        dataset_name = self.config.repo_id.split("/")[1]
        return dest_root / dataset_name / self.config.revision

    def _download(self, filename: str, local_dir: Path) -> Path:
        cfg = self.config
        try:
            result = self._downloader(
                cfg.repo_id,
                filename,
                repo_type=cfg.repo_type,
                revision=cfg.revision,
                local_dir=local_dir,
            )
        except Exception as exc:  # any failure of the external download becomes a QFError
            raise QFError(
                f"download of {filename} from {cfg.repo_id}@{cfg.revision[:12]} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        path = Path(result) if isinstance(result, str | Path) else None
        if path is None or not path.is_file():
            raise QFError(f"download of {filename} returned no file (got {result!r})")
        return path

    def fetch(self, dest_root: Path, run_id: str) -> ArtifactRef:
        target = self.target(dest_root)
        if target.exists():
            raise QFError(f"{target} already exists; verify it or delete it to download again")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=target.parent))
        try:
            payload = staging / "payload"
            payload.mkdir()
            for name in self.config.files:
                shutil.move(self._download(name, staging / "download"), payload / name)
            write_provenance(payload, self.config)
            payload.rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return write_artifact(
            target,
            "raw_dataset",
            RAW_SCHEMA_VERSION,
            run_id,
            extra={"repo_id": self.config.repo_id, "revision": self.config.revision},
            root=self._project_root(),
        )

    def verify(self, ref: ArtifactRef) -> list[str]:
        target = self._project_root() / ref.path
        if not target.is_dir():
            return [f"{ref.path}: directory is missing"]
        problems = verify_provenance(target)
        with contextlib.suppress(QFError):  # an unreadable provenance is already reported above
            problems += _config_problems(read_provenance(target), self.config)
        if artifact_sha256(target) != ref.sha256:
            problems.append(f"{ref.path}: content hash differs from its artifact manifest")
        return problems


@dataclass(frozen=True)
class FetchResult:
    ref: ArtifactRef
    downloaded: bool
    run_dir: Path | None


def raw_dataset_path(source: RawSource, root: Path) -> Path:
    """Where `source` keeps its raw dataset inside the project (under data/raw/)."""
    return source.target(root / DATA_RAW)


def _existing_ref(target: Path, root: Path) -> ArtifactRef:
    return load_artifact_manifest(target, root=root).ref


def verify_raw_dataset(source: RawSource, *, root: Path) -> tuple[ArtifactRef, list[str]]:
    """Check an already fetched raw dataset against its manifest, provenance and config."""
    target = raw_dataset_path(source, root)
    if not target.exists():
        relative = target.relative_to(root)
        raise QFError(f"{relative} does not exist: run `qf data fetch` first")
    ref = _existing_ref(target, root)
    return ref, source.verify(ref)


def fetch_raw_dataset(
    source: RawSource, *, root: Path, source_config: Mapping[str, Any]
) -> FetchResult:
    """Fetch the raw dataset unless it is already present and intact.

    A present but modified dataset is an error: it is never silently re-downloaded.
    A run manifest is written only when a download actually produced a new artifact.
    """
    target = raw_dataset_path(source, root)
    if target.exists():
        ref, problems = verify_raw_dataset(source, root=root)
        if problems:
            listing = "\n".join(f"  - {problem}" for problem in problems)
            raise QFError(f"{ref.path} does not match its provenance:\n{listing}")
        return FetchResult(ref=ref, downloaded=False, run_dir=None)
    run = start_run(_RUN_KIND, root)
    ref = source.fetch(root / DATA_RAW, run.run_id)
    run.finish(
        packages=["huggingface_hub"],
        data_hashes={ref.path.as_posix(): ref.sha256},
        config=dict(source_config),
    )
    return FetchResult(ref=ref, downloaded=True, run_dir=run.run_dir)
