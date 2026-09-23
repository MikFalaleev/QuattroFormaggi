from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from tests.conftest import REPO_ROOT


def _lint_imports() -> str:
    candidate = Path(sys.executable).with_name("lint-imports")
    found = str(candidate) if candidate.exists() else shutil.which("lint-imports")
    assert found, "lint-imports not installed: run `uv sync --extra dev`"
    return found


def test_import_contracts() -> None:
    result = subprocess.run(
        [_lint_imports(), "--no-cache"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout
