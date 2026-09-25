"""Machine-readable log of `qf` commands (D-118): one JSON line per command in
`runs/commands-<machine>.jsonl` — arguments, start, duration, exit code and the run directories
it created. One file per machine (a non-identifying label: OS + short hash of the host name), so
copying a GPU machine's `runs/` never overwrites the Mac's.
`$QF_COMMAND_LOG` points it elsewhere (tests use a temporary file)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from qf.common import RUNS, QFError, machine_label, project_root

__all__ = ["COMMAND_LOG_ENV", "command_log_path", "list_runs", "record_command"]

COMMAND_LOG_ENV: Final = "QF_COMMAND_LOG"


def command_log_path() -> Path | None:
    """`$QF_COMMAND_LOG`, else `runs/commands-<machine>.jsonl` of the project (None outside one)."""
    if os.environ.get(COMMAND_LOG_ENV):
        return Path(os.environ[COMMAND_LOG_ENV])
    try:
        return project_root() / RUNS / f"commands-{machine_label()}.jsonl"
    except QFError:
        return None


def list_runs() -> set[str]:
    try:
        runs = project_root() / RUNS
    except QFError:
        return set()
    return {p.name for p in runs.iterdir() if p.is_dir()} if runs.is_dir() else set()


def record_command(argv: Sequence[str], started: datetime, exit_code: int,
                   runs_before: set[str]) -> None:  # fmt: skip
    path = command_log_path()
    if path is None:
        return
    ended = datetime.now(UTC)
    line = {"started": started.isoformat(), "seconds": round((ended - started).total_seconds(), 3),
            "machine": machine_label(), "argv": list(argv), "exit_code": exit_code,
            "new_runs": sorted(list_runs() - runs_before)}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
