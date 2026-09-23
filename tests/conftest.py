from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty quattro-formaggi project root in tmp_path, selected via $QF_PROJECT_ROOT."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "quattro-formaggi"\n', encoding="utf-8")
    monkeypatch.setenv("QF_PROJECT_ROOT", str(root))
    return root
