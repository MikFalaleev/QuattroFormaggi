"""Bookkeeping files of a training run (D-118): the event log (`events.jsonl`: started,
model_loaded, checkpoint, evaluated, resumed, integrity, samples, completed, stopped, failed —
with time, step and reason), the preflight records (`preflight.jsonl`) and the console copy."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from qf.contracts import Preflight

__all__ = [
    "CONSOLE_FILE",
    "EVENTS_FILE",
    "PREFLIGHT_FILE",
    "append_event",
    "append_preflight",
    "read_events",
    "read_preflights",
]

EVENTS_FILE: Final = "events.jsonl"


def append_event(run_dir: Path, event: str, **fields: Any) -> None:
    line = {"time": datetime.now(UTC).isoformat(), "event": event, **fields}
    with (run_dir / EVENTS_FILE).open("a", encoding="utf-8") as out:
        out.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / EVENTS_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


PREFLIGHT_FILE: Final = "preflight.jsonl"
CONSOLE_FILE: Final = "console.log"


def append_preflight(run_dir: Path, preflight: Preflight) -> None:
    """One line per session: `qf train run` writes it before any model is loaded."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / PREFLIGHT_FILE).open("a", encoding="utf-8") as out:
        out.write(preflight.model_dump_json() + "\n")


def read_preflights(run_dir: Path) -> list[Preflight]:
    path = run_dir / PREFLIGHT_FILE
    if not path.exists():
        return []
    return [Preflight.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line]  # fmt: skip
