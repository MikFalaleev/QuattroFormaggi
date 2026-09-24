"""Scoring, metrics, slices and reports of card_v2 answers (sub-step V6, D-085, D-104…D-106)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qf.backends import FakeBackend
from qf.cli.main import main
from qf.contracts import (
    CaseScoreV2,
    ConflictV2,
    ExtractionTargetV2,
    Length,
    Message,
    OversizeCondition,
    Quantity,
    SecuringCondition,
    SFTRecord,
)
from qf.domain import get_target_schema
from qf.eval import (
    METRICS,
    MOCK_WARNING,
    compare_runs,
    load_run,
    record_kind,
    run_eval,
    score_prediction_v2,
    slice_of,
)
from tests import fake_eval
from tests.factories import make_card_v2, make_record_v2, make_target_v2
from tests.test_harness import BENCH, config, install_bench, variant

SERIALIZE = get_target_schema("card_v2").serialize
Answer = Callable[[dict[str, Any]], None]


def metres(value: float) -> Length:
    return Length(value=value, unit="m")


def gold(name: str) -> ExtractionTargetV2:
    """reefer: +2…+6 °C and a temperature sensor; reefer_no_temp: the temperature is not in the
    text; tent: no conditions; securing_no_values: «нужно крепление»; lowbed_partial: two of
    three dimensions; lowbed: all three dimensions and chains + chocks."""
    lowbed = {"equipment_type": "lowbed", "cargo_category": "machinery", "pieces": 1}
    cards = {
        "reefer": make_card_v2(),
        "reefer_no_temp": make_card_v2(special_conditions=[]),
        "tent": make_card_v2(equipment_type="tent", special_conditions=[]),
        "securing_no_values": make_card_v2(equipment_type="tent", special_conditions=[
            SecuringCondition(kind="securing", methods=[])]),
        "lowbed_partial": make_card_v2(**lowbed, special_conditions=[
            OversizeCondition(kind="oversize", length=metres(9.5), width=None,
                              height=metres(3.6))]),
        "lowbed": make_card_v2(**lowbed, special_conditions=[
            SecuringCondition(kind="securing", methods=["chains", "wheel_chocks"]),
            OversizeCondition(kind="oversize", length=metres(9.5), width=metres(3.2),
                              height=metres(3.6))]),
    }  # fmt: skip
    return make_target_v2(cards[name])


def score(target: ExtractionTargetV2, change: Answer | None = None) -> CaseScoreV2:
    answer = json.loads(SERIALIZE(target))
    if change is not None:
        change(answer)
    return score_prediction_v2("case", target, json.dumps(answer, ensure_ascii=False))


def conditions(answer: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = answer["card"]["special_conditions"]
    return result


def drop_sensors(a: dict[str, Any]) -> None:
    a["card"]["special_conditions"] = [c for c in conditions(a) if c["kind"] != "sensors"]


def add_temperature(low: float | None, high: float | None, *, asked: bool) -> Answer:
    def change(a: dict[str, Any]) -> None:
        conditions(a).insert(0, {"kind": "temperature", "min_c": low, "max_c": high})
        if asked and low is not None:
            a["missing_fields"].remove("special_conditions.temperature")

    return change


def fill_securing(a: dict[str, Any]) -> None:
    conditions(a)[0]["methods"] = ["straps"]
    a["missing_fields"].remove("special_conditions.securing")


def fill_width(a: dict[str, Any]) -> None:
    conditions(a)[0]["width"] = {"value": 3, "unit": "m"}
    a["missing_fields"].remove("special_conditions.oversize")


def warmer(a: dict[str, Any]) -> None:
    conditions(a)[0]["max_c"] = 7


@pytest.mark.parametrize(("gold_name", "change", "critical", "kind", "correct"), [
    ("reefer", drop_sensors, ["missed_condition"], "sensors", False),
    ("reefer_no_temp", add_temperature(2, 6, asked=True), ["hallucinated_required_field"],
     "temperature", False),
    ("reefer_no_temp", add_temperature(None, None, asked=False), [], "temperature", False),
    ("tent", add_temperature(2, 6, asked=False), ["hallucinated_condition"], "temperature",
     False),
    ("securing_no_values", fill_securing, ["hallucinated_required_field"], "securing", False),
    ("lowbed_partial", fill_width, ["hallucinated_required_field"], "oversize", False),
    ("reefer", warmer, [], "temperature", False),
    ("reefer", None, [], "temperature", True),
])  # fmt: skip
def test_each_condition_mistake_gets_one_label(
    gold_name: str, change: Answer | None, critical: list[str], kind: str, correct: bool
) -> None:
    result = score(gold(gold_name), change)
    assert result.critical_errors == critical
    assert result.condition_correct[kind] is correct  # type: ignore[index]
    assert result.field_correct["special_conditions"] is correct
    invented = change is not None and change not in (drop_sensors, warmer)
    assert ("special_conditions" in result.hallucinated_fields) is invented


def test_order_units_and_permutations_do_not_matter() -> None:
    def shuffled(a: dict[str, Any]) -> None:
        conditions(a).reverse()
        for condition in conditions(a):
            if condition["kind"] == "securing":
                condition["methods"].reverse()
            if condition["kind"] == "oversize":
                condition["width"] = {"value": 320.4, "unit": "cm"}  # 3.204 m: within 0.01 m

    result = score(gold("lowbed"), shuffled)
    assert result.critical_errors == [] and result.hallucinated_fields == []
    assert result.field_correct["special_conditions"] is True
    assert result.key_fields_correct is True and result.schema_valid

    def too_wide(a: dict[str, Any]) -> None:
        conditions(a)[1]["width"] = {"value": 330, "unit": "cm"}

    too_far = score(gold("lowbed"), too_wide)
    assert too_far.condition_correct["oversize"] is False and too_far.critical_errors == []


def test_card_fields_are_scored_as_in_card_v1() -> None:
    def pieces_one(a: dict[str, Any]) -> None:
        a["card"]["pieces"] = 1
        a["missing_fields"].remove("pieces")

    invented = score(make_target_v2(make_card_v2(pieces=None)), pieces_one)
    assert invented.critical_errors == ["hallucinated_required_field"]
    assert invented.hallucinated_fields == ["pieces"]
    conflict = make_target_v2(make_card_v2(pieces=None),
                              conflicts=[ConflictV2(field="pieces", values=[14, 15])])  # fmt: skip
    assert score(conflict, pieces_one).critical_errors == ["missed_conflict"]
    per_piece = make_target_v2(make_card_v2(weight_total=None, weight_per_piece=Quantity(
        value=500, unit="kg"), pieces=4))  # fmt: skip
    computed = score(per_piece, lambda a: a["card"].update(
        weight_total={"value": 2000, "unit": "kg"}))  # fmt: skip
    assert computed.notes == ["computed_weight_total"] and computed.hallucinated_fields == []


def test_conditions_are_key_fields() -> None:
    assert score(gold("reefer")).key_fields_correct is True
    assert score(gold("reefer"), warmer).key_fields_correct is False
    assert score(gold("reefer")).conditions_answer == ["temperature", "sensors"]


def test_failed_answer() -> None:
    result = score_prediction_v2("case", gold("reefer"), "{not json")
    assert result.failed and result.conditions_answer == []
    assert result.condition_correct == {
        "temperature": False,
        "securing": None,
        "packaging": None,
        "oversize": None,
        "sensors": False,
    }
    assert result.field_correct["special_conditions"] is False and result.critical_errors == []


def test_condition_metrics_by_hand() -> None:
    scores = [
        score(gold("reefer")),  # temperature, sensors found
        score(gold("reefer"), drop_sensors),  # sensors lost
        score(gold("tent"), add_temperature(2, 6, asked=False)),  # temperature invented
        score(gold("tent")),  # nothing to find, nothing stated
    ]

    def value(name: str) -> float | None:
        metric = METRICS.get(name)()
        return metric.aggregate([metric.value(s) for s in scores])

    assert value("condition_recall") == 3 / 4
    assert value("condition_precision") == 3 / 4
    assert value("condition_recall.sensors") == 1 / 2
    assert value("condition_precision.temperature") == 2 / 3
    assert value("condition_precision.securing") is None
    assert value("special_conditions_exact_rate") == 2 / 4
    assert value("field_accuracy.special_conditions") == 1 / 3
    assert value("field_accuracy.special_conditions.temperature") == 2 / 3
    assert value("critical_error_count.missed_condition") == 1
    assert value("critical_error_count.hallucinated_condition") == 1


# --- kinds of cases and the harness --------------------------------------------------------


def record(n: int, target: ExtractionTargetV2, hard_cases: list[str] | None = None,
           **changes: Any) -> SFTRecord:  # fmt: skip
    base = make_record_v2(SERIALIZE(target))
    messages = [base.messages[0], Message(role="user", content=f"Заявка {n}"),
                base.messages[2]]  # fmt: skip
    return make_record_v2(SERIALIZE(target), id=f"qf-test-LOAD{n:05d}-0",
                          group_id=f"load:LOAD{n:05d}", split="test",
                          variant=variant(hard_cases or []), messages=messages,
                          **changes)  # fmt: skip


def bench_v2() -> list[SFTRecord]:
    return [
        record(0, gold("tent")),
        record(1, gold("reefer")),
        record(2, gold("reefer_no_temp"), ["required_condition_dropped"]),
        record(3, make_target_v2(make_card_v2(pieces=None)), ["dropped_fields"]),
        record(4, gold("lowbed"), template_family="manual_mock"),
    ]


def test_kinds_of_card_v2_cases() -> None:
    kinds = [record_kind(r) for r in bench_v2()]
    assert kinds == ["clean", "conditions", "hard", "missing", "manual"]
    assert [slice_of(r, "hard_case") for r in bench_v2()] == [
        "clean", "clean", "required_condition_dropped", "dropped_fields", "manual"]  # fmt: skip


def test_harness_scores_card_v2(fake_project: Path) -> None:
    records = bench_v2()
    ref = install_bench(fake_project, records)
    answers = {r.messages[1].content: r.messages[2].content for r in records}
    tent = json.loads(answers["Заявка 0"])
    tent["card"]["special_conditions"] = [{"kind": "temperature", "min_c": 2, "max_c": 6}]
    answers["Заявка 0"] = json.dumps(tent, ensure_ascii=False)
    names = ["key_field_accuracy", "special_conditions_exact_rate", "condition_precision",
             "critical_error_count", "critical_error_count.hallucinated_condition"]  # fmt: skip
    cfg = config(ref, metrics=names)
    evaluated = run_eval(records, FakeBackend.from_answers(answers), cfg, bench=ref,
                         root=fake_project)  # fmt: skip
    overall = {name: entry["value"] for name, entry in evaluated.table["overall"].items()}
    assert overall["critical_error_count.hallucinated_condition"] == 1
    assert overall["special_conditions_exact_rate"] == 4 / 5
    assert overall["condition_precision"] == 6 / 7  # 6 kinds found, the tent's one invented
    report = (evaluated.run_dir / "report.md").read_text(encoding="utf-8")
    assert f"> {MOCK_WARNING}." in report
    assert "| Ошибка | clean | conditions | missing | hard | manual | трудные (всё, кроме clean) " \
           "| всего |" in report  # fmt: skip
    assert "| `hallucinated_condition` | 1 | 0 | 0 | 0 | 0 | 0 | 1 |" in report
    assert "`special_conditions`: эталон [] → ответ [" in report
    loaded = load_run(evaluated.run_dir)
    assert all(isinstance(s, CaseScoreV2) for s in loaded.scores)
    assert "B − A" in compare_runs(loaded, loaded)
    assert BENCH.name in evaluated.state["bench_path"]


# --- fake answers of the acceptance run ----------------------------------------------------


@pytest.mark.parametrize(("gold_name", "index", "kind", "critical"), [
    ("securing_no_values", 1, "filled_condition_values", ["hallucinated_required_field"]),
    ("reefer_no_temp", 9, "invented_required_temperature", ["hallucinated_required_field"]),
    ("reefer", 2, "dropped_condition", ["missed_condition"]),
    ("tent", 4, "added_temperature", ["hallucinated_condition"]),
    ("reefer", 6, "wrong_temperature", []),
    ("lowbed", 8, "reordered_same", []),
])  # fmt: skip
def test_fake_condition_mistakes(gold_name: str, index: int, kind: str,
                                 critical: list[str]) -> None:  # fmt: skip
    target = gold(gold_name)
    got, answer = fake_eval.mistake(index, SERIALIZE(target))
    assert got == kind
    result = score_prediction_v2("case", target, answer)
    assert result.critical_errors == critical
    assert (result.field_correct["special_conditions"] is True) is (kind == "reordered_same")


def test_fake_eval_on_a_card_v2_bench(fake_project: Path) -> None:
    records = bench_v2()
    install_bench(fake_project, records)
    assert fake_eval.main(["--bench", str(BENCH)]) == 0
    out = fake_project / fake_eval.DEFAULT_OUT
    for name in ("fake_bench_t.yaml", "fake_bench_t_gold.yaml"):
        assert main(["eval", "run", "--config", str(out / name)]) == 0
