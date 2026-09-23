"""Project root discovery, standard directories and atomic file writes."""

from __future__ import annotations

import contextlib
import os
import tempfile
import tomllib
from pathlib import Path

from qf.common.errors import QFError

__all__ = [
    "ARTIFACTS",
    "CONFIGS",
    "DATA_PROCESSED",
    "DATA_RAW",
    "DATA_SPLITS",
    "PROJECT_NAME",
    "ROOT_ENV_VAR",
    "RUNS",
    "atomic_write_text",
    "project_root",
]

PROJECT_NAME = "quattro-formaggi"
ROOT_ENV_VAR = "QF_PROJECT_ROOT"
# mkstemp creates 0600 files; written files get the usual permissions of a new file instead.
WRITTEN_FILE_MODE = 0o644

# Standard directories, relative to project_root().
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_SPLITS = Path("data/splits")
ARTIFACTS = Path("artifacts")
RUNS = Path("runs")
CONFIGS = Path("configs")


def _is_project_root(directory: Path) -> bool:
    pyproject = directory / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    project = data.get("project")
    return isinstance(project, dict) and project.get("name") == PROJECT_NAME


def _search_upwards(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _is_project_root(candidate):
            return candidate
    return None


def project_root(start: Path | None = None) -> Path:
    """Find the directory whose pyproject.toml declares the quattro-formaggi project.

    Order: explicit `start` (searched upwards) > $QF_PROJECT_ROOT > cwd (upwards) >
    location of this file (upwards).
    """
    if start is not None:
        found = _search_upwards(start)
        if found is None:
            raise QFError(f"no {PROJECT_NAME} pyproject.toml found at or above {start}")
        return found
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        root = Path(override).resolve()
        if not _is_project_root(root):
            raise QFError(f"${ROOT_ENV_VAR}={override} does not contain the {PROJECT_NAME} project")
        return root
    for origin in (Path.cwd(), Path(__file__).parent):
        found = _search_upwards(origin)
        if found is not None:
            return found
    raise QFError(f"cannot find the {PROJECT_NAME} project root; set ${ROOT_ENV_VAR}")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` so readers see either the old or the new content, never a mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, WRITTEN_FILE_MODE)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
