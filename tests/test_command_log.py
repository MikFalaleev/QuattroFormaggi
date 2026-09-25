"""The machine-readable log of `qf` commands (D-118)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qf.cli.command_log import COMMAND_LOG_ENV, command_log_path
from qf.cli.main import main


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_every_command_is_logged(tmp_path: Path) -> None:
    assert main(["export", "merge"]) == 2
    assert main(["train", "fetch-base"]) == 1  # refused: still logged
    logged = lines(tmp_path / "commands.jsonl")
    assert [(entry["argv"], entry["exit_code"]) for entry in logged] == [
        (["export", "merge"], 2),
        (["train", "fetch-base"], 1),
    ]
    assert all(entry["machine"] and entry["seconds"] >= 0 for entry in logged)


def test_default_path_is_per_machine_in_the_project(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(COMMAND_LOG_ENV)
    monkeypatch.setattr("platform.node", lambda: "MacBook-Pro-Someone.local")
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    path = command_log_path()
    assert path is not None and path.parent == fake_project / "runs"
    assert re.fullmatch(r"commands-darwin-[0-9a-f]{6}\.jsonl", path.name)
    (fake_project / "runs" / "20260925-000000-x-aaaaaa").mkdir(parents=True)
    assert main(["export", "gguf"]) == 2
    entry = lines(path)[0]
    assert entry["new_runs"] == [] and "Someone" not in json.dumps(entry)


def test_help_and_version_are_not_logged(tmp_path: Path) -> None:
    assert main(["--version"]) == 0
    assert not (tmp_path / "commands.jsonl").exists()
