"""Contract tests for stage artifacts (IMPLEMENTATION_PLAN.md C.6, C.8)."""

from __future__ import annotations

import shutil
import typing
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from qf.common import (
    ArtifactKind,
    ArtifactRef,
    QFError,
    lineage,
    load_artifact_manifest,
    manifest_path_for,
    read_artifact,
    sha256_dir,
    write_artifact,
)

KINDS: tuple[ArtifactKind, ...] = typing.get_args(ArtifactKind)
PARENT = "a" * 64


def _file(root: Path, relative: str, content: str = '{"id": 1}\n') -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", KINDS)
def test_roundtrip_every_kind(fake_project: Path, kind: ArtifactKind) -> None:
    path = _file(fake_project, "data/processed/x.jsonl")
    ref = write_artifact(Path("data/processed/x.jsonl"), kind, "v1", "run-1", parents=[PARENT])
    assert ref.path == PurePosixPath("data/processed/x.jsonl")
    assert ref.parents == (PARENT,)
    assert manifest_path_for(path).is_file()
    assert read_artifact(path, kind, {"v1"}) == ref


def test_directory_artifact_roundtrip(fake_project: Path) -> None:
    _file(fake_project, "artifacts/adapters/a/adapter_config.json", "{}")
    _file(fake_project, "artifacts/adapters/a/weights/part-1.bin", "x")
    ref = write_artifact(fake_project / "artifacts/adapters/a", "lora_adapter", "peft_lora_v1", "r")
    assert (fake_project / "artifacts/adapters/a.manifest.json").is_file()
    assert read_artifact(Path("artifacts/adapters/a"), "lora_adapter", {"peft_lora_v1"}) == ref


def test_tampered_byte_detected(fake_project: Path) -> None:
    path = _file(fake_project, "data/raw/t.csv", "a,b\n1,2\n")
    write_artifact(path, "raw_dataset", "v1", "r")
    path.write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        read_artifact(path, "raw_dataset", {"v1"})


def test_wrong_kind_rejected(fake_project: Path) -> None:
    path = _file(fake_project, "data/splits/b.jsonl")
    write_artifact(path, "benchmark", "v1", "r")
    with pytest.raises(QFError, match="expected artifact kind 'sft_dataset'"):
        read_artifact(path, "sft_dataset", {"v1"})


def test_unsupported_version_rejected(fake_project: Path) -> None:
    path = _file(fake_project, "data/splits/b.jsonl")
    write_artifact(path, "benchmark", "bench_v2", "r")
    with pytest.raises(QFError, match="unsupported schema version 'bench_v2'"):
        read_artifact(path, "benchmark", {"bench_v1"})


@pytest.mark.parametrize("bad", ["/abs/file.jsonl", "../escape.jsonl", "data/../../x", "."])
def test_ref_path_must_be_relative(bad: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            kind="metrics",
            path=PurePosixPath(bad),
            sha256=PARENT,
            schema_version="v1",
            producer_run_id="r",
        )


def test_artifact_outside_root_rejected(fake_project: Path, tmp_path: Path) -> None:
    outside = _file(tmp_path, "outside.txt")
    with pytest.raises(QFError, match="outside the project root"):
        write_artifact(outside, "metrics", "v1", "r")


def test_invalid_reference_fields_rejected(fake_project: Path) -> None:
    path = _file(fake_project, "runs/r/metrics.json")
    with pytest.raises(QFError, match="invalid artifact reference"):
        write_artifact(path, "metrics", "", "r")


def test_missing_artifact_and_manifest(fake_project: Path) -> None:
    with pytest.raises(QFError, match="artifact does not exist"):
        write_artifact(Path("data/nothing.jsonl"), "metrics", "v1", "r")
    path = _file(fake_project, "data/no_manifest.jsonl")
    with pytest.raises(QFError, match="missing or unreadable artifact manifest"):
        read_artifact(path, "metrics", {"v1"})
    manifest_path_for(path).write_text("{}", encoding="utf-8")
    with pytest.raises(QFError, match="invalid artifact manifest"):
        read_artifact(path, "metrics", {"v1"})


def test_tree_copied_to_another_root_is_readable(tmp_path: Path) -> None:
    root_a, root_b = tmp_path / "machine_a", tmp_path / "machine_b"
    path = _file(root_a, "data/processed/generated_v1/train.jsonl")
    ref = write_artifact(path, "sft_dataset", "sft_record_v1", "r", root=root_a)
    shutil.copytree(root_a, root_b)
    copied = read_artifact(
        Path("data/processed/generated_v1/train.jsonl"),
        "sft_dataset",
        {"sft_record_v1"},
        root=root_b,
    )
    assert copied == ref


def test_moved_artifact_detected(fake_project: Path) -> None:
    path = _file(fake_project, "data/a.jsonl")
    write_artifact(path, "metrics", "v1", "r")
    moved = fake_project / "data" / "b.jsonl"
    path.rename(moved)
    manifest_path_for(path).rename(manifest_path_for(moved))
    with pytest.raises(QFError, match="describes a different location"):
        read_artifact(moved, "metrics", {"v1"})


def test_dir_hash_independent_of_creation_order(tmp_path: Path) -> None:
    names = ["b.txt", "a/z.txt", "a/c.txt", "d.txt"]
    for directory, order in ((tmp_path / "one", names), (tmp_path / "two", names[::-1])):
        for name in order:
            _file(directory, name, f"content of {name}")
    assert sha256_dir(tmp_path / "one") == sha256_dir(tmp_path / "two")
    (tmp_path / "two" / "d.txt").rename(tmp_path / "two" / "e.txt")
    assert sha256_dir(tmp_path / "one") != sha256_dir(tmp_path / "two")


def test_manifest_extra_roundtrip(fake_project: Path) -> None:
    path = _file(fake_project, "runs/r/predictions.jsonl")
    write_artifact(path, "predictions", "predictions_v1", "r", extra={"backend": "fake", "n": 3})
    assert load_artifact_manifest(path).extra == {"backend": "fake", "n": 3}


def test_lineage_follows_parents(fake_project: Path) -> None:
    raw = write_artifact(_file(fake_project, "data/raw/loads.csv", "l"), "raw_dataset", "v1", "r1")
    facts = write_artifact(
        _file(fake_project, "data/processed/load_facts.jsonl", "f"),
        "load_facts",
        "v1",
        "r2",
        parents=[raw.sha256],
    )
    train = write_artifact(
        _file(fake_project, "data/processed/train.jsonl", "t"),
        "sft_dataset",
        "v1",
        "r3",
        parents=[facts.sha256, raw.sha256],
    )
    assert lineage(train) == [facts, raw]
    assert lineage(raw) == []


def test_lineage_missing_parent_raises(fake_project: Path) -> None:
    ref = write_artifact(
        _file(fake_project, "data/processed/x.jsonl"), "sft_dataset", "v1", "r", parents=[PARENT]
    )
    with pytest.raises(QFError, match="not found"):
        lineage(ref)


def test_lineage_reports_corrupt_manifest(fake_project: Path) -> None:
    ref = write_artifact(
        _file(fake_project, "data/processed/x.jsonl"), "sft_dataset", "v1", "r", parents=[PARENT]
    )
    _file(fake_project, "runs/bad/y.json.manifest.json", "not json")
    with pytest.raises(QFError, match="invalid artifact manifest"):
        lineage(ref)
