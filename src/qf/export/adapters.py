"""A backup copy of an accepted LoRA adapter and its verification (plan step 15, D-121).

`backup_adapter` copies `runs/<id>/adapter/` to `artifacts/adapters/<name>/` and writes the
copy's own artifact manifest (the same sha256, producer run and parents; the source in
`extra`); a copy whose hash differs from the source is removed. `verify_adapter` checks an
adapter directory (a backup or the one in a run) against its manifest and its own
`qf_adapter_manifest.json` before it is merged (step 16) or restored from a backup.
Only files are read: no torch, no model.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from qf.common import (
    ArtifactRef,
    QFError,
    load_artifact_manifest,
    manifest_path_for,
    read_artifact,
    write_artifact,
)
from qf.contracts import ModelRef, supported_versions

__all__ = [
    "ADAPTERS_DIR",
    "ADAPTER_MANIFEST",
    "REQUIRED_FILES",
    "TRAINED_LOADER",
    "AdapterVerification",
    "backup_adapter",
    "verify_adapter",
]

ADAPTERS_DIR: Final = Path("artifacts/adapters")
ADAPTER_MANIFEST: Final = "qf_adapter_manifest.json"
"""Written into the adapter directory by `qf train run` (base, revision, data, loader, seed)."""
TRAINED_LOADER: Final = "cuda_4bit"
"""The loader of real training; `tiny_random_cpu` adapters come from the CPU rehearsal."""
REQUIRED_FILES: Final = ("adapter_model.safetensors", "adapter_config.json", ADAPTER_MANIFEST)
_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class AdapterVerification:
    path: str
    sha256: str | None = None
    checks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _adapter_ref(path: Path, root: Path) -> ArtifactRef:
    return read_artifact(path, "lora_adapter", supported_versions("lora_adapter"), root=root)


def backup_adapter(source: Path, name: str, root: Path) -> ArtifactRef:
    """Copy a verified adapter to `artifacts/adapters/<name>/` with its own manifest."""
    if not _NAME.match(name):
        raise QFError(f"backup name {name!r}: use letters, digits, '.', '_' and '-'")
    source_ref = _adapter_ref(source, root)  # the source matches its manifest before copying
    destination = root / ADAPTERS_DIR / name
    if destination.exists() or manifest_path_for(destination).exists():
        raise QFError(f"{destination.relative_to(root)} already exists: choose another name")
    extra: dict[str, Any] = {
        **load_artifact_manifest(root / source_ref.path, root=root).extra,
        "backup_of": source_ref.path.as_posix(),
    }
    partial = destination.with_name(f".{name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(root / source_ref.path, partial)
    partial.rename(destination)
    ref = write_artifact(destination, "lora_adapter", source_ref.schema_version,
                         source_ref.producer_run_id, source_ref.parents, extra=extra,
                         root=root)  # fmt: skip
    if ref.sha256 != source_ref.sha256:
        shutil.rmtree(destination)
        manifest_path_for(destination).unlink(missing_ok=True)
        raise QFError(f"the copy of {source_ref.path} has sha256 {ref.sha256}, the source "
                      f"{source_ref.sha256}: copy removed")  # fmt: skip
    return ref


def _read_json(path: Path, result: AdapterVerification) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.problems.append(f"{path.name}: not readable ({exc})")
        return None
    if not isinstance(loaded, dict):
        result.problems.append(f"{path.name}: not a JSON object")
        return None
    return loaded


def verify_adapter(path: Path, root: Path, *, base: ModelRef | None = None) -> AdapterVerification:
    """Hash against the manifest, the files of a PEFT adapter, the training run it names, the
    base model (`base`: the pinned one) and, for a backup, the hash of its source."""
    result = AdapterVerification(path=str(path))
    try:
        ref = _adapter_ref(path, root)
    except QFError as exc:
        result.problems.append(str(exc))
        return result
    result.sha256 = ref.sha256
    result.checks.append(f"sha256 {ref.sha256[:12]}… matches its manifest")
    directory = root / ref.path
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        result.problems.append(f"missing files: {', '.join(missing)}")
        return result
    result.checks.append(f"files present: {', '.join(REQUIRED_FILES)}")
    own = _read_json(directory / ADAPTER_MANIFEST, result)
    config = _read_json(directory / "adapter_config.json", result)
    if own is None or config is None:
        return result
    if own.get("loader") != TRAINED_LOADER:
        result.problems.append(f"loader {own.get('loader')!r}: not a training run on the base "
                               f"model (expected {TRAINED_LOADER!r})")  # fmt: skip
    else:
        result.checks.append(f"trained with loader {TRAINED_LOADER} ({own.get('quantization')})")
    named = (config.get("base_model_name_or_path"), config.get("revision"))
    if named != (own.get("base_model"), own.get("revision")):
        result.problems.append(f"adapter_config names {named[0]}@{named[1]}, its manifest "
                               f"{own.get('base_model')}@{own.get('revision')}")  # fmt: skip
    trained_on = (own.get("base_model"), own.get("revision"))
    if base is not None and trained_on != (base.id, base.revision):
        result.problems.append(f"trained on {own.get('base_model')}@{own.get('revision')}, "
                               f"not the pinned {base.id}@{base.revision}")  # fmt: skip
    elif base is not None:
        result.checks.append(f"base model {base.id}@{base.revision[:12]} (pinned)")
    if own.get("run_id") != ref.producer_run_id:
        result.problems.append(f"{ADAPTER_MANIFEST} names run {own.get('run_id')}, the artifact "
                               f"manifest {ref.producer_run_id}")  # fmt: skip
    else:
        result.checks.append(f"run {ref.producer_run_id}, experiment {own.get('experiment_id')}, "
                             f"seed {own.get('seed')}, step {own.get('global_step')}")  # fmt: skip
    source = load_artifact_manifest(directory, root=root).extra.get("backup_of")
    if source is not None:
        _check_source(Path(source), ref, root, result)
    return result


def _check_source(source: Path, ref: ArtifactRef, root: Path,
                  result: AdapterVerification) -> None:  # fmt: skip
    """A backup: the same hash as the source adapter, when the source is still on this machine."""
    if not (root / source).exists():
        result.checks.append(f"backup of {source.as_posix()} (the source is not on this machine)")
        return
    try:
        source_ref = _adapter_ref(source, root)
    except QFError as exc:
        result.problems.append(f"source {source.as_posix()}: {exc}")
        return
    if source_ref.sha256 != ref.sha256:
        result.problems.append(f"source {source.as_posix()} has sha256 {source_ref.sha256[:12]}…, "
                               f"the backup {ref.sha256[:12]}…")  # fmt: skip
    else:
        result.checks.append(f"backup of {source.as_posix()}: the same sha256")
