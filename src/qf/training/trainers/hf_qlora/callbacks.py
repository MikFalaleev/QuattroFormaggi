"""Trainer callbacks (plan step 12): stop conditions, the training log and the run manifest.

A stop condition asks the Trainer to save a checkpoint and stop, and records why in the shared
`RunState`; `train.py` then writes the final status (`aborted` or `failed`) to the manifest.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from qf.common import RunStatus, read_manifest, update_manifest
from qf.training import append_event

__all__ = [
    "BatchLog",
    "EventCallback",
    "JsonlLoggerCallback",
    "ManifestCallback",
    "NanGuardCallback",
    "RunState",
    "WallTimeLimitCallback",
]


def _now() -> float:
    """The clock of the wall-time limit (a module function so tests can move time)."""
    return time.monotonic()


@dataclass
class RunState:
    status: RunStatus = "running"
    reason: str | None = None

    def stop(self, status: RunStatus, reason: str) -> None:
        if self.status == "running":
            self.status, self.reason = status, reason


@dataclass
class BatchLog:
    """Records seen by the training forward passes since the last log line."""

    record_ids: list[str]
    pending: list[str] = field(default_factory=list)

    def add(self, indexes: list[int]) -> None:
        self.pending += [self.record_ids[i] for i in indexes]

    def take(self) -> list[str]:
        taken, self.pending = self.pending, []
        return taken


def _save_and_stop(control: TrainerControl) -> None:
    control.should_save = True
    control.should_training_stop = True


class WallTimeLimitCallback(TrainerCallback):
    def __init__(self, max_minutes: float, state: RunState) -> None:
        self.limit_s = max_minutes * 60
        self.state = state
        self.started = _now()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs: Any) -> None:  # fmt: skip
        self.started = _now()

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs: Any) -> None:  # fmt: skip
        if _now() - self.started > self.limit_s and not control.should_training_stop:
            _save_and_stop(control)
            self.state.stop("aborted", "wall_time")


class NanGuardCallback(TrainerCallback):
    """A NaN or infinite loss or gradient norm: checkpoint for diagnosis, stop, status failed."""

    def __init__(self, state: RunState) -> None:
        self.state = state

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: dict[str, Any] | None = None, **kwargs: Any) -> None:  # fmt: skip
        for key in ("loss", "grad_norm", "eval_loss"):
            value = (logs or {}).get(key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                _save_and_stop(control)
                self.state.stop("failed", f"{key} is {value} at step {state.global_step}")
                return


class JsonlLoggerCallback(TrainerCallback):
    """Every log line (train and eval) into `train_log.jsonl`; a train line lists the records
    of its optimizer step."""

    def __init__(self, path: Path, batches: BatchLog) -> None:
        self.path = path
        self.batches = batches

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: dict[str, Any] | None = None, **kwargs: Any) -> None:  # fmt: skip
        line: dict[str, Any] = {"step": state.global_step, "epoch": state.epoch,
                                "time": time.time(), **(logs or {})}  # fmt: skip
        if "loss" in line:
            line["record_ids"] = self.batches.take()
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(line, ensure_ascii=False, allow_nan=True) + "\n")


class ManifestCallback(TrainerCallback):
    """The latest step, loss, learning rate and gradient norm in `run_manifest.json`."""

    KEYS = ("loss", "learning_rate", "grad_norm", "eval_loss")

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.latest: dict[str, Any] = {}

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               logs: dict[str, Any] | None = None, **kwargs: Any) -> None:  # fmt: skip
        values = {k: v for k, v in (logs or {}).items() if k in self.KEYS}
        if not values:
            return
        self.latest.update({"step": state.global_step, **values})
        latest = {f"last_{k}": v for k, v in self.latest.items()}
        metrics = {**read_manifest(self.run_dir).metrics, **latest}
        update_manifest(self.run_dir, metrics=metrics)


class EventCallback(TrainerCallback):
    """Checkpoints and evaluations into `events.jsonl`; the current step to `on_step`."""

    def __init__(self, run_dir: Path, on_step: Callable[[int], None]) -> None:
        self.run_dir = run_dir
        self.on_step = on_step

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs: Any) -> None:  # fmt: skip
        self.on_step(state.global_step)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                **kwargs: Any) -> None:  # fmt: skip
        append_event(self.run_dir, "checkpoint", step=state.global_step,
                     path=f"checkpoint-{state.global_step}")  # fmt: skip

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:  # fmt: skip
        append_event(self.run_dir, "evaluated", step=state.global_step,
                     eval_loss=(metrics or {}).get("eval_loss"))  # fmt: skip
