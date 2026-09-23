from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from qf.common import ROOT_ENV_VAR, QFError, atomic_write_text, project_root
from tests.conftest import REPO_ROOT

paths_module = importlib.import_module("qf.common.paths")


def test_project_root_found_from_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(REPO_ROOT / "src" / "qf")
    assert project_root() == REPO_ROOT


def test_project_root_falls_back_to_package_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert project_root() == REPO_ROOT


def test_project_root_env_override(fake_project: Path) -> None:
    assert project_root() == fake_project.resolve()


def test_project_root_env_must_point_to_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    with pytest.raises(QFError, match=ROOT_ENV_VAR):
        project_root()


def test_project_root_from_start_searches_upwards(fake_project: Path) -> None:
    nested = fake_project / "data" / "raw"
    nested.mkdir(parents=True)
    assert project_root(nested) == fake_project.resolve()


def test_project_root_ignores_other_projects_and_bad_toml(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pyproject.toml").write_text("[project\nname=")
    for start in (other, broken):
        with pytest.raises(QFError, match="no quattro-formaggi"):
            project_root(start)


def test_project_root_not_found_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(paths_module, "_search_upwards", lambda start: None)
    with pytest.raises(QFError, match=ROOT_ENV_VAR):
        project_root()


def test_atomic_write_text_creates_parents_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "второй")
    assert target.read_text(encoding="utf-8") == "второй"
    assert sorted(p.name for p in target.parent.iterdir()) == ["file.txt"]
    assert target.stat().st_mode & 0o777 == 0o644
