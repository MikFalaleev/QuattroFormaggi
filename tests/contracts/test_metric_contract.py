"""Contract test of every registered metric (plan C.8, step 9).

A new metric registered in `METRICS` is checked here without new tests: on perfect answers it
gets its best value, on unparsable answers the worst value or None, and it aggregates an
empty list without failing. The gold cases state every card field at least once (including
temperature_c of a reefer and a per-piece weight) and conflicts of both kinds, in card_v1 and
in card_v2 (every kind of special condition, sub-step V6): the metrics of one version are None
for the scores of the other.
"""

from __future__ import annotations

from datetime import date

import pytest

import qf.cli.wiring  # noqa: F401  (every registry)
from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    CaseScore,
    Conflict,
    ConflictV2,
    ExtractionTarget,
    ExtractionTargetV2,
    Length,
    Metric,
    OversizeCondition,
    PackagingCondition,
    Quantity,
    SecuringCondition,
)
from qf.domain import get_target_schema, serialize_target
from qf.eval import FIELDS, FIELDS_V2, METRICS, score_prediction, score_prediction_v2
from tests.factories import make_card, make_card_v2, make_target, make_target_v2


def kg(value: float) -> Quantity:
    return Quantity(value=value, unit="kg")


def gold_cases() -> list[ExtractionTarget]:
    reefer = make_card(equipment_type="reefer", temperature_c=-18, delivery_date=date(2022, 1, 3))
    per_piece = make_card(weight_total=None, weight_per_piece=kg(500), pieces=4)
    return [
        make_target(reefer),
        make_target(per_piece),
        make_target(make_card(weight_total=None),
                    conflicts=[Conflict(field="weight_total", values=[kg(9000), kg(10500)])]),
        make_target(make_card(pieces=None), conflicts=[Conflict(field="pieces", values=[14, 15])]),
        make_target(make_card(pickup_date=None, shipper_name=None)),
    ]  # fmt: skip


def gold_cases_v2() -> list[ExtractionTargetV2]:
    def metres(value: float) -> Length:
        return Length(value=value, unit="m")

    lowbed = make_card_v2(
        equipment_type="lowbed",
        cargo_category="machinery",
        pieces=1,
        special_conditions=[
            SecuringCondition(kind="securing", methods=["chains"]),
            OversizeCondition(
                kind="oversize", length=metres(9.5), width=metres(3.2), height=metres(3.6)
            ),
        ],
    )
    van = make_card_v2(
        equipment_type="van",
        special_conditions=[PackagingCondition(kind="packaging", types=["crate"])],
    )
    tent = make_card_v2(equipment_type="tent", special_conditions=[], weight_total=None)
    return [
        make_target_v2(),  # a reefer with a temperature and a temperature sensor
        make_target_v2(lowbed),
        make_target_v2(van),
        make_target_v2(make_card_v2(weight_total=None, weight_per_piece=kg(500), pieces=4)),
        make_target_v2(tent, conflicts=[ConflictV2(field="weight_total",
                                                   values=[kg(9000), kg(10500)])]),
        make_target_v2(make_card_v2(pieces=None),
                       conflicts=[ConflictV2(field="pieces", values=[14, 15])]),
    ]  # fmt: skip


def test_gold_cases_state_every_field() -> None:
    cases = gold_cases()
    assert all(any(getattr(t.card, f) is not None for t in cases) for f in FIELDS)
    assert {c.field for t in cases for c in t.conflicts} == {"weight_total", "pieces"}
    cases_v2 = gold_cases_v2()
    assert all(any(getattr(t.card, f) not in (None, []) for t in cases_v2) for f in FIELDS_V2)
    kinds = {c.kind for t in cases_v2 for c in t.card.special_conditions}
    assert kinds == set(CARD_V2_CONDITION_KINDS)
    assert {c.field for t in cases_v2 for c in t.conflicts} == {"weight_total", "pieces"}


def scores(raw: str | None = None) -> list[CaseScore]:
    """Perfect answers to the gold cases of both versions, or the same `raw` answer to each."""
    serialize_v2 = get_target_schema("card_v2").serialize
    return [
        *(score_prediction(f"case-{i}", gold, serialize_target(gold) if raw is None else raw)
          for i, gold in enumerate(gold_cases())),
        *(score_prediction_v2(f"case-v2-{i}", gold, serialize_v2(gold) if raw is None else raw)
          for i, gold in enumerate(gold_cases_v2())),
    ]  # fmt: skip


def aggregate(metric: Metric, cases: list[CaseScore]) -> float | None:
    return metric.aggregate([metric.value(c) for c in cases])


@pytest.mark.parametrize("name", METRICS.names())
def test_metric_is_registered_under_its_name(name: str) -> None:
    metric = METRICS.get(name)()
    assert isinstance(metric, Metric)
    assert metric.name == name
    assert isinstance(metric.higher_is_better, bool)


@pytest.mark.parametrize("name", METRICS.names())
def test_perfect_answers_get_the_best_value(name: str) -> None:
    metric = METRICS.get(name)()
    value = aggregate(metric, scores())
    assert value is not None, "the gold cases must exercise every metric"
    assert value == (1.0 if metric.higher_is_better else 0.0)


@pytest.mark.parametrize("name", METRICS.names())
def test_invalid_json_gets_the_worst_value_or_none(name: str) -> None:
    metric = METRICS.get(name)()
    perfect = aggregate(metric, scores())
    broken = aggregate(metric, scores("{not json"))
    assert perfect is not None
    if broken is not None:
        assert broken < perfect if metric.higher_is_better else broken > perfect


@pytest.mark.parametrize("name", METRICS.names())
def test_aggregate_of_nothing_is_safe(name: str) -> None:
    metric = METRICS.get(name)()
    assert metric.aggregate([]) is None
    assert metric.aggregate([None, None]) is None
