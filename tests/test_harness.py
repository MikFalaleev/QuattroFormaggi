"""The eval harness, reports and `qf eval` commands on the fake backend (plan step 9)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from qf.backends import TIMEOUT, FakeBackend
from qf.cli.main import main
from qf.common import (
    ArtifactRef,
    ComponentConfig,
    QFError,
    atomic_write_text,
    read_artifact,
    read_manifest,
    sha256_json,
    sha256_text,
    write_artifact,
)
from qf.contracts import (
    Conflict,
    ExtractionTarget,
    GenerationRequest,
    GenerationResult,
    Message,
    Quantity,
    SFTRecord,
    VariantInfo,
)
from qf.domain import get_target_schema, serialize_target
from qf.eval import (
    FAKE_BANNER,
    METRICS_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    SCORES_FILE,
    SYNTHETIC_WARNING,
    BenchRef,
    BootstrapSettings,
    EvalConfig,
    EvalRun,
    GenerationSettings,
    compare_runs,
    compare_stage,
    load_bench,
    load_run,
    record_kind,
    run_eval,
)
from tests.factories import make_card, make_record, make_record_v2, make_target

SYSTEM = "system prompt"
BENCH = Path("data/splits/bench_t.jsonl")
METRIC_NAMES = ["json_valid_rate", "key_field_accuracy", "failed_output_rate",
                "hallucination_rate", "critical_error_count", "missing_f1",
                "critical_error_count.hallucinated_required_field"]  # fmt: skip


def variant(hard_cases: list[str], ood: Any = None) -> VariantInfo:
    return VariantInfo(weight_unit="kg", weight_mode="total", dropped_fields=[],
                       hard_cases=hard_cases, request_date=date(2021, 12, 28),
                       date_style="iso", city_lang="ru", ood_reason=ood)  # fmt: skip


def targets() -> list[tuple[ExtractionTarget, list[str], Any]]:
    kg = Quantity(value=12000, unit="kg")
    conflict = Conflict(field="weight_total", values=[kg, Quantity(value=13, unit="t")])
    return [
        (make_target(), [], None),
        (make_target(make_card(pieces=None)), ["dropped_fields"], "route"),
        (make_target(make_card(weight_total=None), conflicts=[conflict]), ["conflict_weight"],
         None),
        (make_target(make_card(equipment_type="reefer")), [], "family"),
    ]  # fmt: skip


def bench_records() -> list[SFTRecord]:
    records = []
    for i, (target, hard_cases, ood) in enumerate(targets()):
        messages = [Message(role="system", content=SYSTEM),
                    Message(role="user", content=f"Заявка номер {i}"),
                    Message(role="assistant", content=serialize_target(target))]  # fmt: skip
        records.append(make_record(id=f"qf-test-LOAD{i:05d}-0", group_id=f"load:LOAD{i:05d}",
                                   split="test", variant=variant(hard_cases, ood),
                                   language="ru" if i % 2 else "en",
                                   messages=messages))  # fmt: skip
    return records


def install_bench(root: Path, records: list[SFTRecord]) -> ArtifactRef:
    path = root / BENCH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    ref = write_artifact(path, "benchmark", "sft_record_v1", "run-bench", root=root)
    atomic_write_text(path.with_suffix(".sha256"), f"{ref.sha256}  {path.name}\n")
    return ref


def config(ref: ArtifactRef, **changes: Any) -> EvalConfig:
    fields: dict[str, Any] = {
        "name": "fake_test", "backend": ComponentConfig(name="fake"),
        "bench": BenchRef(path=BENCH, sha256=ref.sha256), "metrics": METRIC_NAMES,
        "bootstrap": BootstrapSettings(n=200, seed=0),
    }  # fmt: skip
    fields.update(changes)
    return EvalConfig(**fields)


def gold_answers(records: list[SFTRecord]) -> dict[str, str]:
    return {r.messages[1].content: r.messages[2].content for r in records}


@pytest.fixture
def root(fake_project: Path) -> Path:
    return fake_project


@pytest.fixture
def bench(root: Path) -> tuple[ArtifactRef, list[SFTRecord]]:
    records = bench_records()
    return install_bench(root, records), records


def run(root: Path, bench: tuple[ArtifactRef, list[SFTRecord]], backend: FakeBackend,
        resume: Path | None = None, **changes: Any) -> EvalRun:  # fmt: skip
    ref, records = bench
    return run_eval(records, backend, config(ref, **changes), bench=ref, root=root, resume=resume)


def test_harness_never_sends_assistant_message(root: Path, bench: Any) -> None:
    backend = FakeBackend.from_answers(gold_answers(bench[1]))
    result = run(root, bench, backend)
    assert len(backend.requests) == len(bench[1])
    for request in backend.requests:
        assert [m.role for m in request.messages] == ["system", "user"]
        assert request.max_tokens == 768 and request.temperature == 0.0
    assert result.table["overall"]["key_field_accuracy"]["value"] == 1.0
    assert result.table["overall"]["json_valid_rate"]["ci"] == [1.0, 1.0]


def test_timeout_recorded_as_failure(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1]) | {"Заявка номер 0": TIMEOUT}
    result = run(root, bench, FakeBackend.from_answers(answers))
    lines = (result.run_dir / PREDICTIONS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(bench[1])  # the case is recorded, not skipped
    first = json.loads(lines[0])
    assert first["id"] == "qf-test-LOAD00000-0" and first["error"] == "timeout"
    assert result.scores[0].failed and result.scores[0].generation_error == "timeout"
    overall = result.table["overall"]
    assert overall["failed_output_rate"]["value"] == pytest.approx(1 / len(bench[1]))
    assert overall["key_field_accuracy"]["value"] == pytest.approx(0.5)  # 1 of 2 applicable


def test_backend_exception_recorded_as_failure(root: Path, bench: Any) -> None:
    class Broken(FakeBackend):
        def generate(self, req: Any) -> Any:
            raise RuntimeError("socket closed")

    result = run(root, bench, Broken(FakeBackend.Config()))
    assert all(s.failed for s in result.scores)
    assert result.predictions["qf-test-LOAD00000-0"]["error"].startswith("backend raised")


def test_resume_skips_done(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    first = run(root, bench, FakeBackend.from_answers(answers))
    path = first.run_dir / PREDICTIONS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    # interrupted while writing the third answer
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][:15], encoding="utf-8")
    backend = FakeBackend.from_answers(answers)
    resumed = run(root, bench, backend, resume=first.run_dir)
    assert [r.messages[1].content for r in backend.requests] == ["Заявка номер 2",
                                                                 "Заявка номер 3"]  # fmt: skip
    assert resumed.run_dir == first.run_dir
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(bench[1])
    assert resumed.table["overall"] == first.table["overall"]


class ServerDown(FakeBackend):
    """The same answers (and model id) as a FakeBackend, but the connection fails once."""

    def generate(self, req: GenerationRequest) -> GenerationResult:
        if req.messages[-1].content == "Заявка номер 1" and not getattr(self, "failed", False):
            self.failed = True
            return GenerationResult(text="", latency_s=0.0, error="connection refused")
        return super().generate(req)


def test_resume_asks_again_where_generation_failed(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    down = ServerDown(FakeBackend.from_answers(answers).config)
    first = run(root, bench, down)
    assert first.scores[1].failed and first.scores[1].generation_error == "connection refused"
    backend = FakeBackend.from_answers(answers)
    resumed = run(root, bench, backend, resume=first.run_dir)
    assert [r.messages[1].content for r in backend.requests] == ["Заявка номер 1"]
    assert not resumed.scores[1].failed
    assert load_run(first.run_dir).predictions["qf-test-LOAD00001-0"]["error"] is None


def test_resume_after_a_complete_line_without_newline(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    first = run(root, bench, FakeBackend.from_answers(answers))
    path = first.run_dir / PREDICTIONS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]), encoding="utf-8")  # no newline after the second
    run(root, bench, FakeBackend.from_answers(answers), resume=first.run_dir)
    assert [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()] == [
        r.id for r in bench[1]
    ]


def test_resume_refuses_a_changed_setup(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    first = run(root, bench, FakeBackend.from_answers(answers))
    with pytest.raises(QFError, match="generation changed"):
        run(root, bench, FakeBackend.from_answers(answers), resume=first.run_dir,
            generation=GenerationSettings(max_tokens=100))  # fmt: skip
    with pytest.raises(QFError, match="model_id changed"):
        run(root, bench, FakeBackend.from_answers(answers, model="other"), resume=first.run_dir)
    with pytest.raises(QFError, match="model_id changed"):  # the fake got other answers
        run(root, bench, FakeBackend.from_answers({}), resume=first.run_dir)
    with pytest.raises(QFError, match="not a run directory"):
        run(root, bench, FakeBackend.from_answers(answers), resume=root / "elsewhere")


def test_corrupt_predictions_line_is_an_error(root: Path, bench: Any) -> None:
    first = run(root, bench, FakeBackend.from_answers(gold_answers(bench[1])))
    path = first.run_dir / PREDICTIONS_FILE
    path.write_text("not json\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(QFError, match="line 1 is not a prediction"):
        run(root, bench, FakeBackend.from_answers(gold_answers(bench[1])), resume=first.run_dir)


def test_manifest_contains_bench_hash_and_prompt_hash(root: Path, bench: Any) -> None:
    ref, _ = bench
    result = run(root, bench, FakeBackend.from_answers(gold_answers(bench[1])))
    manifest = read_manifest(result.run_dir)
    assert manifest.kind == "eval" and manifest.status == "completed"
    assert manifest.data_hashes[BENCH.as_posix()] == ref.sha256
    assert manifest.prompt_hashes == {"system": sha256_text(SYSTEM)}
    assert manifest.config["backend_name"] == "fake"
    assert manifest.config["model_id"].startswith("fake:fake@")  # + hash of the answers
    assert manifest.config["generation"] == {"max_tokens": 768, "temperature": 0.0,
                                             "json_schema": False}  # fmt: skip
    assert result.state["generation"] == {"max_tokens": 768, "temperature": 0.0}  # as in step 9
    assert manifest.hardware["ram_total_bytes"] > 0 and manifest.hardware["machine"]
    assert manifest.metrics["key_field_accuracy"] == 1.0
    predictions = read_artifact(result.run_dir / PREDICTIONS_FILE, "predictions",
                                {"predictions_v1"}, root=root)  # fmt: skip
    assert predictions.parents == (ref.sha256,)
    metrics = read_artifact(result.run_dir / METRICS_FILE, "metrics", {"metrics_v1"}, root=root)
    assert metrics.parents == (ref.sha256, predictions.sha256)
    assert (result.run_dir / SCORES_FILE).exists()


def test_unknown_metric_fails_before_any_request(root: Path, bench: Any) -> None:
    backend = FakeBackend.from_answers(gold_answers(bench[1]))
    with pytest.raises(QFError, match="unknown metric 'nope'"):
        run(root, bench, backend, metrics=["json_valid_rate", "nope"])
    assert backend.requests == []


def test_config_rejects_slice_metric_outside_metrics(bench: Any) -> None:
    with pytest.raises(ValueError, match="slice_metrics not in metrics"):
        config(bench[0], slice_metrics=["missing_f1", "conflict_recall"])
    with pytest.raises(ValueError, match="repeated"):
        config(bench[0], metrics=["missing_f1", "missing_f1"])


def test_load_bench_checks_hashes(root: Path, bench: Any) -> None:
    ref, records = bench
    loaded_ref, loaded = load_bench(config(ref), root)
    assert loaded_ref == ref and [r.id for r in loaded] == [r.id for r in records]
    with pytest.raises(QFError, match="differs from the config"):
        load_bench(config(ref, bench=BenchRef(path=BENCH, sha256="0" * 64)), root)
    (root / BENCH).with_suffix(".sha256").write_text("f" * 64 + "  x\n", encoding="utf-8")
    with pytest.raises(QFError, match="differs"):
        load_bench(config(ref), root)


def test_repeated_record_ids_are_refused(root: Path, bench: Any) -> None:
    ref, records = bench
    with pytest.raises(QFError, match="repeated ids"):
        run_eval([*records, records[0]], FakeBackend.from_answers({}), config(ref), bench=ref,
                 root=root)  # fmt: skip


def test_answers_without_a_scorer_are_refused(
    root: Path, bench: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, records = bench
    backend = FakeBackend.from_answers({})
    monkeypatch.setattr("qf.eval.harness.SCORED_SCHEMAS", ("card_v1",))
    with pytest.raises(QFError, match="card_v2 cannot be scored"):
        run_eval([make_record_v2()], backend, config(ref), bench=ref, root=root)
    with pytest.raises(QFError, match="different answer schemas"):
        run_eval([records[0], make_record_v2()], backend, config(ref), bench=ref, root=root)
    assert backend.requests == []


def test_mixed_system_prompts_are_refused(root: Path, bench: Any) -> None:
    ref, records = bench
    other = records[0].model_copy(update={"messages": [Message(role="system", content="other"),
                                                       *records[0].messages[1:]]})  # fmt: skip
    with pytest.raises(QFError, match="different system prompts"):
        run_eval([other, *records[1:]], FakeBackend.from_answers({}), config(ref), bench=ref,
                 root=root)  # fmt: skip


def test_slices_in_metrics(root: Path, bench: Any) -> None:
    assert [record_kind(r) for r in bench[1]] == ["clean", "missing", "hard", "clean"]
    result = run(root, bench, FakeBackend.from_answers(gold_answers(bench[1])))
    slices = result.table["slices"]
    assert set(slices["kind"]) == {"clean", "missing", "hard"}
    assert slices["kind"]["clean"]["cases"] == 2
    assert set(slices["distribution"]) == {"in_dist", "ood"}
    assert set(slices["ood_reason"]) == {"—", "route", "family"}
    assert set(slices["source"]) == {"synthetic"}
    assert "ci" not in slices["kind"]["clean"]["missing_f1"]


def wrong_answers(records: list[SFTRecord]) -> dict[str, str]:
    """Gold answers with typical mistakes: a code fence, an origin/destination swap, invented
    pieces and invalid JSON."""
    answers = gold_answers(records)
    first = make_target()
    swapped = make_target(make_card(origin=first.card.destination,
                                    destination=first.card.origin))  # fmt: skip
    answers["Заявка номер 0"] = serialize_target(swapped)
    invented = make_target(make_card(pieces=1), missing_fields=[])
    answers["Заявка номер 1"] = f"```json\n{serialize_target(invented)}\n```"
    answers["Заявка номер 3"] = '{"card": '
    return answers


def test_report_is_readable(root: Path, bench: Any) -> None:
    result = run(root, bench, FakeBackend.from_answers(wrong_answers(bench[1])))
    report = (result.run_dir / REPORT_FILE).read_text(encoding="utf-8")
    assert FAKE_BANNER in report and SYNTHETIC_WARNING in report
    for heading in ("## Метрики", "## Критические ошибки по видам кейсов", "## Срезы",
                    "### kind", "## Худшие кейсы"):  # fmt: skip
        assert heading in report
    # columns: clean, missing, hard, the hard slice of the gates (missing + hard), total
    assert "| `origin_destination_swap` | 1 | 0 | 0 | 0 | 1 |" in report
    assert "| `hallucinated_required_field` | 0 | 1 | 0 | 1 | 1 |" in report
    assert "| не оценено | 1 | 0 | 0 | 0 | 1 |" in report  # the invalid JSON (a clean case)
    # the worst case first: the unparsable answer, then the critical errors
    worst = report.index("## Худшие кейсы")
    assert report.index("qf-test-LOAD00003-0", worst) < report.index("qf-test-LOAD00000-0", worst)
    assert "`origin`: эталон Пермь (Пермский край) → ответ Самара (Самарская область)" in report
    assert "строгий разбор не прошёл (format)" in report
    assert "Заявка номер 1" in report
    # the fenced answer sits inside a longer fence and does not break the report
    assert "````text\n```json\n{" in report
    assert report.count("````text") == 1 and report.count("\n````\n") == 1


def test_compare_runs(root: Path, bench: Any) -> None:
    a = run(root, bench, FakeBackend.from_answers(wrong_answers(bench[1])))
    b = run(root, bench, FakeBackend.from_answers(gold_answers(bench[1]), model="gold"))
    text = compare_runs(load_run(a.run_dir), load_run(b.run_dir))
    assert FAKE_BANNER in text
    assert "Разные backend" not in text
    assert "| `key_field_accuracy` | 0 | 1 | 1 |" in text
    # case 0 (swap) and case 3 (invalid JSON) have key fields; cases 1 and 2 do not
    assert "Исправлено (A неверно, B верно): 2; сломано (A верно, B неверно): 0" in text
    assert "- **Исправлено** (2): `qf-test-LOAD00000-0`, `qf-test-LOAD00003-0`" in text
    assert "- **Сломано** (0): —" in text


def test_compare_allows_other_backend_but_not_other_bench(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    a = load_run(run(root, bench, FakeBackend.from_answers(answers)).run_dir)
    b = load_run(run(root, bench, FakeBackend.from_answers(answers)).run_dir)
    other_backend = EvalRun(b.run_dir, {**b.state, "backend": "openai_local"}, [],
                            b.predictions, b.scores, b.table)  # fmt: skip
    assert "Разные backend" in compare_runs(a, other_backend)
    for key in ("bench_sha256", "prompt_sha256", "schema_version"):
        changed = EvalRun(b.run_dir, {**b.state, key: "x"}, [], b.predictions, b.scores,
                          b.table)  # fmt: skip
        with pytest.raises(QFError, match=f"{key} differs"):
            compare_runs(a, changed)
    fewer = EvalRun(b.run_dir, b.state, [], b.predictions, b.scores[:-1], b.table)
    with pytest.raises(QFError, match="different cases"):
        compare_runs(a, fewer)


def test_compare_stage_writes_a_run(root: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    a = run(root, bench, FakeBackend.from_answers(answers))
    b = run(root, bench, FakeBackend.from_answers(answers))
    path, text = compare_stage(a.run_dir, b.run_dir, root=root)
    assert path.read_text(encoding="utf-8") == text
    manifest = read_manifest(path.parent)
    assert manifest.kind == "eval-compare"
    assert manifest.config["run_a"] == a.run_id
    with pytest.raises(QFError, match="not a finished eval run"):
        compare_stage(a.run_dir, root / "runs" / "missing", root=root)


def test_cli_eval_run_and_compare(
    root: Path, bench: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    ref, records = bench
    responses = {sha256_text(q): a for q, a in wrong_answers(records).items()}
    (root / "responses.json").write_text(json.dumps(responses, ensure_ascii=False))
    cfg = config(ref).model_dump(mode="json")
    cfg["backend"] = {"name": "fake", "responses_file": "responses.json"}
    (root / "eval.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert main(["eval", "run", "--config", str(root / "eval.yaml")]) == 0
    out = capsys.readouterr().out
    assert "FAKE backend" in out and "4 cases, 1 failed" in out
    run_dirs = sorted(p for p in (root / "runs").iterdir() if "-eval-" in p.name)
    assert main(["eval", "run", "--config", str(root / "eval.yaml"),
                 "--resume", str(run_dirs[-1])]) == 0  # fmt: skip
    assert main(["eval", "compare", str(run_dirs[-1]), str(run_dirs[-1])]) == 0
    assert "Comparison:" in capsys.readouterr().out
    cfg["metrics"] = ["nope"]
    (root / "bad.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    assert main(["eval", "run", "--config", str(root / "bad.yaml")]) == 1


def test_fake_backend_rejects_bad_responses_file(root: Path) -> None:
    (root / "bad.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(QFError, match="expected a JSON object"):
        FakeBackend(FakeBackend.Config(responses_file=Path("bad.json")))
    with pytest.raises(QFError, match="cannot read fake responses"):
        FakeBackend(FakeBackend.Config(responses_file=root / "absent.json"))


def test_json_schema_mode(root: Path, bench: Any) -> None:
    """Constrained decoding (step 10): the answer schema goes with every request and is part of
    the run's setup; without it the state keeps the form of the step 9 runs."""
    ref, records = bench
    backend = FakeBackend.from_answers(gold_answers(records))
    on = GenerationSettings(json_schema=True)
    result = run(root, bench, backend, generation=on)
    schema = get_target_schema("card_v1").json_schema()
    assert backend.requests and all(r.json_schema == schema for r in backend.requests)
    assert result.state["generation"] == {"max_tokens": 768, "temperature": 0.0,
                                          "json_schema": True}  # fmt: skip
    assert result.state["json_schema_sha256"] == sha256_json(schema)
    with pytest.raises(QFError, match="generation, json_schema_sha256 changed"):
        run(root, bench, FakeBackend.from_answers(gold_answers(records)), resume=result.run_dir)
    off = run(root, bench, FakeBackend.from_answers(gold_answers(records)))
    assert "json_schema_sha256" not in off.state and "json_schema" not in off.state["generation"]
    assert "json_schema True" in (result.run_dir / REPORT_FILE).read_text(encoding="utf-8")


def test_cli_eval_baseline(root: Path, bench: Any, capsys: pytest.CaptureFixture[str]) -> None:
    ref, records = bench
    responses = {sha256_text(q): a for q, a in gold_answers(records).items()}
    (root / "responses.json").write_text(json.dumps(responses, ensure_ascii=False))
    cfg = config(ref).model_dump(mode="json")
    cfg["backend"] = {"name": "fake", "responses_file": "responses.json"}
    (root / "base.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    for mode, suffix in (("on", "schema"), ("off", "free")):
        before = set((root / "runs").iterdir()) if (root / "runs").exists() else set()
        assert main(["eval-baseline", "--config", str(root / "base.yaml"),
                     "--json-schema", mode]) == 0  # fmt: skip
        (newest,) = set((root / "runs").iterdir()) - before
        state = json.loads((newest / "eval_state.json").read_text(encoding="utf-8"))
        assert state["name"] == f"fake_test.{suffix}"
        assert state["generation"].get("json_schema", False) is (mode == "on")
    assert main(["eval-baseline", "--config", str(root / "base.yaml")]) == 2  # the mode is required
    capsys.readouterr()
