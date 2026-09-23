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


# Tests with these markers touch the outside world or special hardware. They are skipped
# unless explicitly enabled, so a bare `pytest` never downloads anything (rule A.2.4).
OPT_IN_MARKERS = {
    "network": "--run-network",
    "gpu": "--run-gpu",
    "lmstudio": "--run-lmstudio",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, flag in OPT_IN_MARKERS.items():
        parser.addoption(flag, action="store_true", help=f"run tests marked '{marker}'")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, flag in OPT_IN_MARKERS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"'{marker}' test: enable with {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
