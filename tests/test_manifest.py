from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qf.common import (
    MANIFEST_FILENAME,
    QFError,
    RunManifest,
    collect_git_info,
    collect_package_versions,
    new_run_id,
    read_manifest,
    update_manifest,
    write_manifest,
)


def _manifest(run_id: str = "20260923-101500-test-abcdef") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        kind="test",
        created_at=datetime(2026, 9, 23, 10, 15, tzinfo=UTC),
        seed=42,
        data_hashes={"train.jsonl": "0" * 64},
        config={"lr": 0.0001},
    )


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    manifest = _manifest()
    path = write_manifest(manifest, tmp_path / "run")
    assert path == tmp_path / "run" / MANIFEST_FILENAME
    assert read_manifest(tmp_path / "run") == manifest


def test_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    write_manifest(_manifest(), run_dir)
    original = (run_dir / MANIFEST_FILENAME).read_text()

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        write_manifest(_manifest().model_copy(update={"seed": 7}), run_dir)
    monkeypatch.undo()
    assert (run_dir / MANIFEST_FILENAME).read_text() == original
    assert [p.name for p in run_dir.iterdir()] == [MANIFEST_FILENAME]  # no temp leftovers


def test_update_manifest_merges_and_validates(tmp_path: Path) -> None:
    write_manifest(_manifest(), tmp_path)
    updated = update_manifest(tmp_path, status="completed", metrics={"loss": 0.5})
    assert updated.status == "completed"
    assert read_manifest(tmp_path).metrics == {"loss": 0.5}
    with pytest.raises(QFError, match="invalid manifest update"):
        update_manifest(tmp_path, status="exploded")
    with pytest.raises(QFError, match="invalid manifest update"):
        update_manifest(tmp_path, not_a_field=1)
    with pytest.raises(QFError, match="cannot be changed"):
        update_manifest(tmp_path, run_id="other")


def test_read_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(QFError, match="cannot read"):
        read_manifest(tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_text('{"run_id": 1}')
    with pytest.raises(QFError, match="invalid run manifest"):
        read_manifest(tmp_path)


def test_git_info_outside_repo_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert collect_git_info(tmp_path) == (None, False)


def test_git_info_inside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q", str(tmp_path)], check=True)
    assert collect_git_info(tmp_path) == (None, False)  # no commits yet
    (tmp_path / "f.txt").write_text("x")
    assert collect_git_info(tmp_path) == (None, True)
    subprocess.run([*git, "-C", str(tmp_path), "add", "f.txt"], check=True)
    subprocess.run([*git, "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    commit, dirty = collect_git_info(tmp_path)
    assert commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit)
    assert dirty is False


def test_git_info_without_git_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # no `git` anywhere on PATH
    assert collect_git_info(tmp_path) == (None, False)


def test_run_id_format() -> None:
    run_id = new_run_id("train")
    assert re.fullmatch(r"\d{8}-\d{6}-train-[0-9a-f]{6}", run_id)
    assert new_run_id("train") != new_run_id("train")


def test_run_id_rejects_bad_kind() -> None:
    with pytest.raises(QFError, match="invalid run kind"):
        new_run_id("Bad Kind")


def test_collect_package_versions() -> None:
    versions = collect_package_versions(["pydantic", "definitely-not-installed-qf-xyz"])
    assert versions["pydantic"] is not None
    assert versions["definitely-not-installed-qf-xyz"] is None
