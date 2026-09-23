"""An eval run as stored in `runs/<run_id>/` (plan step 9): read by the report and the
comparison, written by the harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from qf.common import QFError, atomic_write_text
from qf.contracts import CaseScore, SFTRecord

__all__ = [
    "EVAL_STATE_FILE",
    "METRICS_FILE",
    "PREDICTIONS_FILE",
    "REPORT_FILE",
    "SCORES_FILE",
    "EvalRun",
    "load_run",
    "read_predictions",
]

EVAL_STATE_FILE: Final = "eval_state.json"
PREDICTIONS_FILE: Final = "predictions.jsonl"
SCORES_FILE: Final = "scores.jsonl"
METRICS_FILE: Final = "metrics.json"
REPORT_FILE: Final = "report.md"


@dataclass(frozen=True)
class EvalRun:
    run_dir: Path
    # name, backend, model_id, bench path and sha256, schema, prompt sha256, generation settings,
    # metric and slice names
    state: dict[str, Any]
    records: list[SFTRecord]  # empty when the run is read back from disk
    predictions: dict[str, dict[str, Any]]  # record id -> {id, output, latency_s, error, ...}
    scores: list[CaseScore]
    table: dict[str, Any]  # the content of metrics.json

    @property
    def run_id(self) -> str:
        return self.run_dir.name


def _prediction(line: str) -> dict[str, Any] | None:
    try:
        item = json.loads(line)
    except ValueError:
        return None
    if (
        isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and {"output", "error"} <= set(item)
    ):
        return item
    return None


def read_predictions(path: Path, *, repair: bool = False) -> dict[str, dict[str, Any]]:
    """Answers already written, by record id; a later line of the same record wins (a resumed
    run asks again where generation failed). A truncated last line (a run interrupted
    mid-write) is an error, or with `repair` is removed so that its case is asked again;
    `repair` also ends the file with a newline, so the next answer starts on a new line."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    done: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(lines, start=1):
        item = _prediction(line)
        if item is None:
            if repair and number == len(lines) and not text.endswith("\n"):
                lines = lines[:-1]
                break
            raise QFError(f"{path}: line {number} is not a prediction")
        done[item["id"]] = item
    if repair and text and not text.endswith("\n"):
        atomic_write_text(path, "".join(f"{kept}\n" for kept in lines))
    return done


def load_run(run_dir: Path) -> EvalRun:
    """A finished run from disk (for `qf eval compare`)."""
    try:
        state = json.loads((run_dir / EVAL_STATE_FILE).read_text(encoding="utf-8"))
        table = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
        lines = (run_dir / SCORES_FILE).read_text(encoding="utf-8").splitlines()
        scores = [CaseScore.model_validate_json(line) for line in lines]
    except (OSError, ValueError) as exc:
        raise QFError(f"{run_dir} is not a finished eval run: {exc}") from exc
    predictions = read_predictions(run_dir / PREDICTIONS_FILE)
    return EvalRun(run_dir, state, [], predictions, scores, table)
