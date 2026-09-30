"""Release of v0.1 (step 19): the release manifest and `release_check`.

A release directory (`artifacts/release/`) holds `release_manifest.json` — the artifacts of the
release with their sha256 — and the model cards. `release_check` verifies that everything the
card talks about exists and matches: the artifacts and their manifests, the lineage of the GGUF
down to the raw dataset (every link is read and its hash checked), the cards (no placeholders,
the hash of every artifact named, a gates table with actual numbers) and the specifications.
Nothing is uploaded anywhere: publishing is the human's decision (plan step 19).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qf.common import (
    QFError,
    StrictConfig,
    artifact_sha256,
    atomic_write_text,
    lineage,
    load_artifact_manifest,
    project_root,
    read_artifact,
)
from qf.contracts import supported_versions

__all__ = [
    "PLACEHOLDER",
    "RELEASE_MANIFEST",
    "ReleaseArtifact",
    "ReleaseConfig",
    "ReleaseManifest",
    "build_release",
    "release_check",
]

RELEASE_MANIFEST: Final = "release_manifest.json"
PLACEHOLDER: Final = re.compile(r"<[^<>\n]{1,80}>")
"""Any `<…>` text in a card is an unfilled placeholder (write commands with UPPERCASE names)."""
_GATE_ROW: Final = re.compile(r"^\|\s*\**\s*([PП])\s*(\d[мm]?)\b[^|]*\|(.*)$", re.MULTILINE)
"""Gates table: one row per gate, `| P1 threshold | measured value | verdict |` (П in the Russian
card); the cells after the first must hold an actual number."""
_NUMBER: Final = re.compile(r"\d")
_REQUIRED_GATES: Final = ("1", "2", "3", "4", "5", "6")
_REQUIRED_KINDS: Final = ("lora_adapter", "sft_dataset", "raw_dataset")
_SHA_PREFIX: Final = 12


class ReleaseArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(pattern=r"^[a-z0-9_]+$")  # adapter | merged | gguf | gguf_bf16 | ...
    path: str
    kind: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release: str
    license: str
    base_model: str
    base_revision: str
    artifacts: list[ReleaseArtifact] = Field(min_length=1)
    cards: list[str] = Field(min_length=1)
    docs: list[str] = Field(min_length=1)


class ReleaseConfig(StrictConfig):
    """`configs/release/release_v0.1.yaml`: what goes into the release (paths relative to the
    project root)."""

    release: str
    license: str
    base_model: str
    base_revision: str
    release_dir: Path = Path("artifacts/release")
    artifacts: dict[str, Path] = Field(min_length=1)  # role -> artifact path
    card_sources: dict[str, Path] = Field(min_length=1)  # card file name -> tracked source
    docs: list[str] = Field(min_length=1)


def _load(release_dir: Path) -> ReleaseManifest:
    path = release_dir / RELEASE_MANIFEST
    try:
        return ReleaseManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QFError(f"missing or unreadable {path}: {exc}") from exc
    except ValidationError as exc:
        raise QFError(f"invalid {path}: {exc}") from exc


def build_release(
    release_dir: Path,
    *,
    release: str,
    license_id: str,
    base_model: str,
    base_revision: str,
    artifacts: Mapping[str, Path],
    card_sources: Mapping[str, Path],
    docs: list[str],
    root: Path | None = None,
) -> ReleaseManifest:
    """Copy the cards into `release_dir` and write the release manifest from the artifacts'
    own manifests (their hashes are read, never recomputed here: `release_check` does that)."""
    root = root if root is not None else project_root()
    entries = []
    for role, path in artifacts.items():
        ref = load_artifact_manifest(path, root=root).ref
        entries.append(ReleaseArtifact(role=role, path=str(ref.path), kind=ref.kind,
                                       sha256=ref.sha256))  # fmt: skip
    release_dir.mkdir(parents=True, exist_ok=True)
    for name, source in card_sources.items():
        shutil.copyfile(source, release_dir / name)
    manifest = ReleaseManifest(release=release, license=license_id, base_model=base_model,
                               base_revision=base_revision, artifacts=entries,
                               cards=list(card_sources), docs=docs)  # fmt: skip
    atomic_write_text(release_dir / RELEASE_MANIFEST,
                      json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False,
                                 indent=2) + "\n")  # fmt: skip
    return manifest


def _check_artifact(entry: ReleaseArtifact, root: Path, verify_hashes: bool) -> list[str]:
    path = root / entry.path
    if not path.exists():
        return [f"{entry.role}: {entry.path} is missing"]
    try:
        manifest_ref = load_artifact_manifest(path, root=root).ref
    except QFError as exc:
        return [f"{entry.role}: {exc}"]
    if manifest_ref.sha256 != entry.sha256:
        return [f"{entry.role}: the release manifest names sha256 {entry.sha256[:_SHA_PREFIX]}… "
                f"but the artifact manifest has {manifest_ref.sha256[:_SHA_PREFIX]}…"]  # fmt: skip
    if verify_hashes:
        actual = artifact_sha256(path)
        if actual != entry.sha256:
            return [f"{entry.role}: content hash mismatch (expected {entry.sha256}, "
                    f"actual {actual})"]  # fmt: skip
    return []


def _check_lineage(gguf: ReleaseArtifact, root: Path, verify_hashes: bool) -> list[str]:
    """Every ancestor of the GGUF is found through the manifests on disk, exists and (with
    hashes) matches; the chain must reach the adapter, the training data and the raw dataset."""
    problems = []
    try:
        ref = load_artifact_manifest(root / gguf.path, root=root).ref
        ancestors = lineage(ref, root=root)
    except QFError as exc:
        return [f"lineage of {gguf.path}: {exc}"]
    kinds = {a.kind for a in ancestors}
    for kind in _REQUIRED_KINDS:
        if kind not in kinds:
            problems.append(f"lineage of {gguf.path}: no '{kind}' ancestor")
    for ancestor in ancestors:
        location = root / ancestor.path
        if not location.exists():
            problems.append(f"lineage: {ancestor.kind} {ancestor.path} is missing")
        elif verify_hashes:
            try:
                read_artifact(location, ancestor.kind, supported_versions(ancestor.kind),
                              root=root)  # fmt: skip
            except QFError as exc:
                problems.append(f"lineage: {exc}")
    return problems


def _check_card(name: str, text: str, manifest: ReleaseManifest) -> list[str]:
    problems = [f"{name}: placeholder {m.group(0)!r}" for m in PLACEHOLDER.finditer(text)]
    for entry in manifest.artifacts:
        if entry.sha256[:_SHA_PREFIX] not in text:
            problems.append(f"{name}: the sha256 of '{entry.role}' "
                            f"({entry.sha256[:_SHA_PREFIX]}…) is not in the card")  # fmt: skip
    rows = {m.group(2).replace("m", "м"): m.group(3) for m in _GATE_ROW.finditer(text)}
    for gate in _REQUIRED_GATES:
        row = rows.get(gate)
        if row is None:
            problems.append(f"{name}: no row for gate {gate} in the gates table")
        elif not _NUMBER.search(row):
            problems.append(f"{name}: the row of gate {gate} holds no actual number")
    return problems


def release_check(
    release_dir: Path, *, root: Path | None = None, verify_hashes: bool = True
) -> list[str]:
    """Problems of the release in `release_dir` (empty list = complete). With
    `verify_hashes=False` only the manifests are compared, not the content."""
    root = root if root is not None else project_root()
    try:
        manifest = _load(release_dir)
    except QFError as exc:
        return [str(exc)]
    problems: list[str] = []
    for entry in manifest.artifacts:
        problems += _check_artifact(entry, root, verify_hashes)
    gguf = next((a for a in manifest.artifacts if a.role == "gguf"), None)
    if gguf is None:
        problems.append("the release has no artifact with the role 'gguf'")
    elif (root / gguf.path).exists():
        problems += _check_lineage(gguf, root, verify_hashes)
    for card in manifest.cards:
        path = release_dir / card
        if not path.is_file():
            problems.append(f"{card}: the model card is missing in {release_dir}")
        else:
            problems += _check_card(card, path.read_text(encoding="utf-8"), manifest)
    for doc in manifest.docs:
        if not (root / doc).is_file():
            problems.append(f"{doc}: the specification is missing")
    return problems
