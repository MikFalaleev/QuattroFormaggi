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
from typing import Any, Final, Literal, get_args

from pydantic import Field, model_validator

from qf.common import (
    RUNS,
    ArtifactRef,
    ComponentConfig,
    QFError,
    RunRecorder,
    StrictConfig,
    atomic_write_text,
    collect_hardware,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_text,
    start_run,
    write_artifact,
)
from qf.contracts import (
    BatchGenerationBackend,
    CaseScore,
    GenerationBackend,
    GenerationRequest,
    GenerationResult,
    Metric,
    MetricValue,
    SFTRecord,
    supported_versions,
)
from qf.domain import get_target_schema, parse_sft_jsonl
from qf.eval.metrics import METRICS, SCORERS, get_scorer
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
    "batch_size_of",
    "load_bench",
    "metrics_table",
    "run_eval",
]

PREDICTIONS_VERSION: Final = "predictions_v1"
METRICS_VERSION: Final = "metrics_v1"
COMPARE_FILE: Final = "compare.md"
SCORED_SCHEMAS: Final = tuple(SCORERS.names())
"""Answer schemas with a registered scorer (card_v1; card_v2 since sub-step V6)."""


class GenerationSettings(StrictConfig):
    max_tokens: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    # constrained decoding: the JSON schema of the answer goes with every request (step 10)
    json_schema: bool = False

    def state(self) -> dict[str, Any]:
        """The settings as recorded in `eval_state.json`: `json_schema` only when on, so the
        states (and reports) of the runs made before step 10 keep their exact form."""
        recorded = self.model_dump()
        if not self.json_schema:
            recorded.pop("json_schema")
        return recorded


class BenchRef(StrictConfig):
    path: Path  # relative to the project root
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # a frozen benchmark, or a whole split such as test / test_ood (plan step 14, D-120)
    kind: Literal["benchmark", "sft_dataset"] = "benchmark"


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
    """The benchmark, verified against its manifest, its `.sha256` file and the config; a
    split (`kind: sft_dataset`) against its manifest and the config."""
    kind = cfg.bench.kind
    ref = read_artifact(cfg.bench.path, kind, supported_versions(kind), root=root)
    sha_file = (root / ref.path).with_suffix(".sha256")
    if kind == "benchmark":
        declared = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.exists() else None
    else:
        declared = ref.sha256  # splits have no `.sha256` file; the manifest hash is checked
    if ref.sha256 != cfg.bench.sha256 or declared != ref.sha256:
        raise QFError(f"{ref.path}: sha256 {ref.sha256} differs from the config "
                      f"({cfg.bench.sha256}) or {sha_file.name} ({declared})")  # fmt: skip
    records, issues = parse_sft_jsonl((root / ref.path).read_text(encoding="utf-8"),
                                      source=ref.path.name,
                                      split="bench" if kind == "benchmark" else None)  # fmt: skip
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


def _generate_batch(backend: BatchGenerationBackend,
                    reqs: Sequence[GenerationRequest]) -> list[GenerationResult]:  # fmt: skip
    started = time.monotonic()
    try:
        results = backend.generate_batch(reqs)
        if len(results) == len(reqs):
            return list(results)
        error = f"backend returned {len(results)} results for {len(reqs)} requests"
    except Exception as exc:  # every case of the batch still counts, as a failure
        error = f"backend raised {type(exc).__name__}: {exc}"
    latency = (time.monotonic() - started) / len(reqs)
    return [GenerationResult(text="", latency_s=latency, error=error) for _ in reqs]


def batch_size_of(backend: GenerationBackend) -> int:
    """How many requests the harness sends at once: 1 unless the backend answers batches."""
    return backend.batch_size if isinstance(backend, BatchGenerationBackend) else 1


def _request(record: SFTRecord, settings: GenerationSettings,
             json_schema: dict[str, Any] | None) -> GenerationRequest:  # fmt: skip
    messages = record.messages[:2]
    if [m.role for m in messages] != ["system", "user"]:
        raise QFError(f"{record.id}: expected system and user messages")
    return GenerationRequest(messages=messages, max_tokens=settings.max_tokens,
                             temperature=settings.temperature,
                             json_schema=json_schema)  # fmt: skip


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
        raise QFError(f"answers in {schema} cannot be scored "
                      f"(scored: {', '.join(SCORED_SCHEMAS)})")  # fmt: skip
    state: dict[str, Any] = {
        "name": cfg.name, "backend": backend.name, "model_id": backend.model_id(),
        "bench_path": bench.path.as_posix(), "bench_sha256": bench.sha256,
        "schema_version": schema,
        "prompt_sha256": _single(records, "system prompts",
                                 lambda r: sha256_text(r.messages[0].content)),
        "generation": cfg.generation.state(), "metrics": cfg.metrics,
        "slice_metrics": cfg.slice_metrics or cfg.metrics, "slices": cfg.slices,
    }  # fmt: skip
    if cfg.generation.json_schema:
        state["json_schema_sha256"] = sha256_json(get_target_schema(schema).json_schema())
    if isinstance(backend, BatchGenerationBackend):  # only then, so older states keep their form
        state["batch_size"] = backend.batch_size
    return state


# What must not change when a run is resumed: otherwise its answers come from two setups.
RESUME_KEYS: Final = (
    "bench_sha256", "prompt_sha256", "schema_version", "generation", "json_schema_sha256",
    "backend", "model_id", "batch_size",
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
    target = get_target_schema(state["schema_version"])
    json_schema = target.json_schema() if cfg.generation.json_schema else None
    unknown = set(done) - {r.id for r in records}
    if unknown:
        raise QFError(f"{predictions_path}: answers to records outside the benchmark, "
                      f"e.g. {sorted(unknown)[0]}")  # fmt: skip
    # a resumed run keeps the answers and asks again where generation failed (e.g. the server
    # went down): the later line of a record wins when predictions are read
    pending = [r for r in records if not (r.id in done and done[r.id]["error"] is None)]
    size = batch_size_of(backend)
    with predictions_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            reqs = [_request(r, cfg.generation, json_schema) for r in chunk]
            results = (_generate_batch(backend, reqs) if isinstance(backend, BatchGenerationBackend)
                       and size > 1 else [_generate(backend, reqs[0])])  # fmt: skip
            for record, result in zip(chunk, results, strict=True):
                item = {"id": record.id, "output": result.text,
                        **result.model_dump(exclude={"text"})}  # fmt: skip
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                done[record.id] = item
            out.flush()
    scorer = get_scorer(state["schema_version"])
    scores = [
        scorer.score(r.id, target.parse(r.messages[2].content), done[r.id]["output"],
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
        hardware=collect_hardware(),
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
