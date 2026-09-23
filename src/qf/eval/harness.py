"""The evaluation harness (plan step 9): benchmark records + any backend -> scores and metrics.

The backend gets exactly `messages[:2]` (system + user) of each record, never the gold answer.
Every answer is written to `predictions.jsonl` as soon as it arrives, so an interrupted run
continues with `--resume`; a run with a different benchmark, prompt or generation settings
is refused. Generation errors are recorded as failed cases, never skipped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, get_args

from pydantic import Field, model_validator

from qf.common import (
    RUNS,
    ArtifactRef,
    ComponentConfig,
    QFError,
    RunRecorder,
    StrictConfig,
    atomic_write_text,
    read_artifact,
    sha256_file,
    sha256_text,
    start_run,
    write_artifact,
)
from qf.contracts import (
    CaseScore,
    GenerationBackend,
    GenerationRequest,
    GenerationResult,
    Metric,
    MetricValue,
    SFTRecord,
    supported_versions,
)
from qf.domain import parse_sft_jsonl, parse_target
from qf.eval.metrics import METRICS, score_prediction
from qf.eval.report import compare_runs, render_report
from qf.eval.results import (
    EVAL_STATE_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    SCORES_FILE,
    EvalRun,
    load_run,
    read_predictions,
)
from qf.eval.slices import SliceName, slice_of
from qf.eval.stats import resample_indices, statistic_ci

__all__ = [
    "METRICS_VERSION",
    "PREDICTIONS_VERSION",
    "COMPARE_FILE",
    "RESUME_KEYS",
    "SCORED_SCHEMAS",
    "BenchRef",
    "BootstrapSettings",
    "EvalConfig",
    "compare_stage",
    "GenerationSettings",
    "load_bench",
    "metrics_table",
    "run_eval",
]

PREDICTIONS_VERSION: Final = "predictions_v1"
METRICS_VERSION: Final = "metrics_v1"
COMPARE_FILE: Final = "compare.md"
SCORED_SCHEMAS: Final = ("card_v1",)
"""Answer schemas `score_prediction` understands; card_v2 is added in sub-step V6 (D-086)."""


class GenerationSettings(StrictConfig):
    max_tokens: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)


class BenchRef(StrictConfig):
    path: Path  # relative to the project root
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BootstrapSettings(StrictConfig):
    n: int = Field(default=2000, ge=100)
    seed: int = 0


class EvalConfig(StrictConfig):
    """`configs/eval/<name>.yaml`: backend, benchmark, generation settings and metric names."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    backend: ComponentConfig
    bench: BenchRef
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    metrics: list[str] = Field(min_length=1)
    # columns of the per-slice tables of the report; by default every metric of the run
    slice_metrics: list[str] | None = None
    slices: list[SliceName] = Field(default_factory=lambda: list(get_args(SliceName)))
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)

    @model_validator(mode="after")
    def _slice_metrics_are_run_metrics(self) -> EvalConfig:
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics: repeated names")
        extra = sorted(set(self.slice_metrics or ()) - set(self.metrics))
        if extra:
            raise ValueError(f"slice_metrics not in metrics: {', '.join(extra)}")
        return self


def load_bench(cfg: EvalConfig, root: Path) -> tuple[ArtifactRef, list[SFTRecord]]:
    """The benchmark, verified against its manifest, its `.sha256` file and the config."""
    ref = read_artifact(cfg.bench.path, "benchmark", supported_versions("benchmark"), root=root)
    sha_file = (root / ref.path).with_suffix(".sha256")
    declared = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.exists() else None
    if ref.sha256 != cfg.bench.sha256 or declared != ref.sha256:
        raise QFError(f"{ref.path}: sha256 {ref.sha256} differs from the config "
                      f"({cfg.bench.sha256}) or {sha_file.name} ({declared})")  # fmt: skip
    records, issues = parse_sft_jsonl((root / ref.path).read_text(encoding="utf-8"),
                                      source=ref.path.name, split="bench")  # fmt: skip
    if issues:
        raise QFError(f"{ref.path}: {len(issues)} invalid record(s), e.g. {issues[0]}")
    return ref, records


def _single(records: Sequence[SFTRecord], what: str, key: Callable[[SFTRecord], str]) -> str:
    """The one value of `key` shared by all records (runs are compared per prompt and schema)."""
    values = {key(r) for r in records}
    if len(values) != 1:
        raise QFError(f"benchmark records use {len(values)} different {what}")
    return values.pop()


# --- generation ----------------------------------------------------------------------------


def _generate(backend: GenerationBackend, req: GenerationRequest) -> GenerationResult:
    started = time.monotonic()
    try:
        return backend.generate(req)
    except Exception as exc:  # a backend must not raise; if it does, the case still counts
        return GenerationResult(text="", latency_s=time.monotonic() - started,
                                error=f"backend raised {type(exc).__name__}: {exc}")  # fmt: skip


def _request(record: SFTRecord, settings: GenerationSettings) -> GenerationRequest:
    messages = record.messages[:2]
    if [m.role for m in messages] != ["system", "user"]:
        raise QFError(f"{record.id}: expected system and user messages")
    return GenerationRequest(messages=messages, max_tokens=settings.max_tokens,
                             temperature=settings.temperature)  # fmt: skip


# --- metrics -------------------------------------------------------------------------------


def _metric_entry(metric: Metric, values: Sequence[MetricValue],
                  indices: Sequence[Sequence[int]] | None) -> dict[str, Any]:  # fmt: skip
    entry: dict[str, Any] = {
        "value": metric.aggregate(values),
        "n": sum(v is not None for v in values),
        "higher_is_better": metric.higher_is_better,
    }
    if indices is not None:
        interval = statistic_ci(values, metric.aggregate, indices)
        entry["ci"] = list(interval) if interval else None
    return entry


def metrics_table(
    records: Sequence[SFTRecord], scores: Sequence[CaseScore], metric_names: Sequence[str],
    slice_names: Sequence[str], bootstrap: BootstrapSettings,
) -> dict[str, Any]:  # fmt: skip
    """Overall metrics with bootstrap intervals and per-slice values (no intervals)."""
    metrics = {name: METRICS.get(name)() for name in metric_names}
    values = {name: [m.value(s) for s in scores] for name, m in metrics.items()}
    indices = resample_indices(len(scores), bootstrap.n, bootstrap.seed)
    overall = {name: _metric_entry(m, values[name], indices) for name, m in metrics.items()}
    slices: dict[str, dict[str, Any]] = {}
    for slice_name in slice_names:
        groups: dict[str, list[int]] = {}
        for i, record in enumerate(records):
            groups.setdefault(slice_of(record, slice_name), []).append(i)
        slices[slice_name] = {
            group: {"cases": len(positions),
                    **{name: _metric_entry(m, [values[name][i] for i in positions], None)
                       for name, m in metrics.items()}}
            for group, positions in sorted(groups.items())
        }  # fmt: skip
    return {"cases": len(scores), "bootstrap": bootstrap.model_dump(), "overall": overall,
            "slices": slices}  # fmt: skip


# --- the run -------------------------------------------------------------------------------


def _state(cfg: EvalConfig, backend: GenerationBackend, bench: ArtifactRef,
           records: Sequence[SFTRecord]) -> dict[str, Any]:  # fmt: skip
    if len({r.id for r in records}) != len(records):
        raise QFError("benchmark records have repeated ids")
    schema = _single(records, "answer schemas", lambda r: r.schema_version)
    if schema not in SCORED_SCHEMAS:
        scored = ", ".join(SCORED_SCHEMAS)
        raise QFError(f"answers in {schema} cannot be scored yet (scored: {scored}); "
                      "card_v2 scoring arrives in sub-step V6")  # fmt: skip
    return {
        "name": cfg.name, "backend": backend.name, "model_id": backend.model_id(),
        "bench_path": bench.path.as_posix(), "bench_sha256": bench.sha256,
        "schema_version": schema,
        "prompt_sha256": _single(records, "system prompts",
                                 lambda r: sha256_text(r.messages[0].content)),
        "generation": cfg.generation.model_dump(), "metrics": cfg.metrics,
        "slice_metrics": cfg.slice_metrics or cfg.metrics, "slices": cfg.slices,
    }  # fmt: skip


# What must not change when a run is resumed: otherwise its answers come from two setups.
RESUME_KEYS: Final = (
    "bench_sha256", "prompt_sha256", "schema_version", "generation", "backend", "model_id",
)  # fmt: skip


def _recorder(root: Path, resume: Path | None, state: Mapping[str, Any]) -> RunRecorder:
    if resume is None:
        return start_run("eval", root)
    if resume.resolve() != (root / RUNS / resume.name).resolve():
        raise QFError(f"cannot resume {resume}: not a run directory under {root / RUNS}")
    try:
        stored = json.loads((resume / EVAL_STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QFError(f"cannot resume {resume}: no readable {EVAL_STATE_FILE} ({exc})") from exc
    changed = [k for k in RESUME_KEYS if stored.get(k) != state.get(k)]
    if changed:
        raise QFError(f"cannot resume {resume.name}: {', '.join(changed)} changed")
    return RunRecorder(kind="eval", run_id=resume.name, root=root, created_at=datetime.now(UTC),
                       started_monotonic=time.monotonic())  # fmt: skip


def run_eval(
    records: Sequence[SFTRecord], backend: GenerationBackend, cfg: EvalConfig, *,
    bench: ArtifactRef, root: Path, resume: Path | None = None,
) -> EvalRun:  # fmt: skip
    """Ask the backend about every record, score the answers, write predictions, scores,
    metrics, a report and the run manifest into `runs/<run_id>/`. With `resume` (a run
    directory) the cases already answered without an error are not asked again."""
    for name in cfg.metrics:
        METRICS.get(name)  # an unknown metric fails before any request is sent
    state = _state(cfg, backend, bench, records)
    run = _recorder(root, resume, state)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run.run_dir / EVAL_STATE_FILE, json.dumps(state, indent=2) + "\n")
    predictions_path = run.run_dir / PREDICTIONS_FILE
    done = read_predictions(predictions_path, repair=True)
    unknown = set(done) - {r.id for r in records}
    if unknown:
        raise QFError(f"{predictions_path}: answers to records outside the benchmark, "
                      f"e.g. {sorted(unknown)[0]}")  # fmt: skip
    with predictions_path.open("a", encoding="utf-8") as out:
        for record in records:
            # a resumed run keeps the answers and asks again where generation failed (e.g. the
            # server went down): the later line of a record wins when predictions are read
            if record.id in done and done[record.id]["error"] is None:
                continue
            result = _generate(backend, _request(record, cfg.generation))
            item = {"id": record.id, "output": result.text, **result.model_dump(exclude={"text"})}
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            done[record.id] = item
    scores = [
        score_prediction(r.id, parse_target(r.messages[2].content), done[r.id]["output"],
                         generation_error=done[r.id]["error"])
        for r in records
    ]  # fmt: skip
    table = metrics_table(records, scores, cfg.metrics, cfg.slices, cfg.bootstrap)
    atomic_write_text(run.run_dir / SCORES_FILE,
                      "".join(s.model_dump_json() + "\n" for s in scores))  # fmt: skip
    predictions_ref = write_artifact(predictions_path, "predictions", PREDICTIONS_VERSION,
                                     run.run_id, [bench.sha256], root=root)  # fmt: skip
    metrics_path = run.run_dir / METRICS_FILE
    atomic_write_text(metrics_path, json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    metrics_ref = write_artifact(metrics_path, "metrics", METRICS_VERSION, run.run_id,
                                 [bench.sha256, predictions_ref.sha256], root=root)  # fmt: skip
    evaluated = EvalRun(run.run_dir, state, list(records), done, scores, table)
    atomic_write_text(run.run_dir / REPORT_FILE, render_report(evaluated))
    run.finish(
        config={**cfg.model_dump(mode="json"), "backend_name": state["backend"],
                "model_id": state["model_id"]},
        data_hashes={bench.path.as_posix(): bench.sha256,
                     predictions_ref.path.as_posix(): predictions_ref.sha256,
                     metrics_ref.path.as_posix(): metrics_ref.sha256},
        prompt_hashes={"system": state["prompt_sha256"]},
        metrics={name: entry["value"] for name, entry in table["overall"].items()},
    )  # fmt: skip
    return evaluated


def compare_stage(run_a: Path, run_b: Path, *, root: Path) -> tuple[Path, str]:
    """`qf eval compare`: the comparison of two finished runs, written to
    `runs/<run_id>/compare.md` with its own run manifest. Returns (path, markdown)."""
    a, b = load_run(run_a), load_run(run_b)
    text = compare_runs(a, b)
    run = start_run("eval-compare", root)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    out = run.run_dir / COMPARE_FILE
    atomic_write_text(out, text)
    run.finish(
        config={"run_a": a.run_id, "run_b": b.run_id,
                "backends": [a.state["backend"], b.state["backend"]],
                "model_ids": [a.state["model_id"], b.state["model_id"]]},
        data_hashes={f"{r.run_id}/{METRICS_FILE}": sha256_file(r.run_dir / METRICS_FILE)
                     for r in (a, b)},
        prompt_hashes={"system": a.state["prompt_sha256"]},
    )  # fmt: skip
    return out, text
