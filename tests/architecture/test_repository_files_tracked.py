from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.conftest import REPO_ROOT

TRACKED_SUFFIXES = {".py", ".yaml", ".yml", ".txt", ".toml", ".md", ".gitkeep"}
TRACKED_DIRS = ("src", "tests", "configs", "docs")


def _repository_files() -> list[str]:
    files = []
    for directory in TRACKED_DIRS:
        for path in (REPO_ROOT / directory).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in TRACKED_SUFFIXES or path.name in TRACKED_SUFFIXES:
                files.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(files)


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="needs a git checkout"
)
def test_no_source_file_is_git_ignored() -> None:
    """Regression: an unanchored `data/` pattern once hid the package src/qf/data/ from git."""
    files = _repository_files()
    assert "src/qf/data/__init__.py" in files
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(files),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.split() == [], f"git-ignored project files: {result.stdout.split()}"
