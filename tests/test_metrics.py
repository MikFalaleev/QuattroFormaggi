"""Scoring of one answer and the registered metrics (plan step 9)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from qf.contracts import CaseScore, Conflict, ExtractionTarget, Place, Quantity
from qf.domain import serialize_target
from qf.eval import (
    CRITICAL_ERRORS,
    FIELDS,
    METRICS,
    card_mass_kg,
    score_prediction,
    values_match,
)
from tests.factories import make_card, make_target


def answer_of(target: ExtractionTarget) -> str:
    return serialize_target(target)


def score(gold: ExtractionTarget, answer: ExtractionTarget | str, **kw: Any) -> CaseScore:
    raw = answer if isinstance(answer, str) else answer_of(answer)
    return score_prediction("case-1", gold, raw, **kw)


def aggregate(name: str, cases: Sequence[CaseScore]) -> float | None:
    metric = METRICS.get(name)()
    return metric.aggregate([metric.value(c) for c in cases])


def kg(value: float) -> Quantity:
    return Quantity(value=value, unit="kg")


def test_perfect_prediction_scores_1() -> None:
    gold = make_target()
    case = score(gold, gold)
    assert case.json_parsed and case.schema_valid and case.json_parsed_lenient
    assert not case.failed
    assert all(v is True for v in case.field_correct.values() if v is not None)
    assert case.key_fields_correct is True
    assert case.missing_exact and case.missing_fp == case.missing_fn == 0
    assert case.hallucinated_fields == [] and case.critical_errors == []
    for name in ("json_valid_rate", "key_field_accuracy", "field_accuracy_micro", "missing_f1",
                 "missing_exact_rate"):  # fmt: skip
        assert aggregate(name, [case]) == 1.0
    for name in ("hallucination_rate", "critical_error_count", "failed_output_rate"):
        assert aggregate(name, [case]) == 0.0


def test_unit_equivalence_t_vs_kg() -> None:
    gold = make_target(make_card(weight_total=kg(12600)))
    answer = make_target(make_card(weight_total=Quantity(value=12.6, unit="t")))
    case = score(gold, answer)
    assert case.field_correct["weight_total"] is True
    assert case.key_fields_correct is True


def test_lb_vs_kg_equivalent_within_tolerance() -> None:
    gold = make_target(make_card(weight_total=kg(12592)))
    close = make_target(make_card(weight_total=Quantity(value=27760, unit="lb")))  # 12591.76 kg
    far = make_target(make_card(weight_total=Quantity(value=27762, unit="lb")))  # 12592.67 kg
    assert score(gold, close).field_correct["weight_total"] is True
    assert score(gold, far).field_correct["weight_total"] is False
    assert values_match("weight_total", kg(100), kg(100.5))
    assert not values_match("weight_total", kg(100), kg(100.6))


def test_pieces_weight_swap_flagged() -> None:
    gold = make_target(make_card(pieces=22, weight_total=kg(12592)))
    swapped = make_target(make_card(pieces=12592, weight_total=kg(22)))
    assert "pieces_weight_swap" in score(gold, swapped).critical_errors
    # only the weight took the pieces number
    weight_only = make_target(make_card(pieces=22, weight_total=kg(22)))
    assert "pieces_weight_swap" in score(gold, weight_only).critical_errors


def test_one_piece_of_one_tonne_is_not_a_swap() -> None:
    gold = make_target(make_card(pieces=1, weight_total=Quantity(value=1, unit="t")))
    assert score(gold, gold).critical_errors == []
    as_kg = make_target(make_card(pieces=1, weight_total=kg(1000)))
    assert score(gold, as_kg).critical_errors == []


def test_wrong_unit_magnitude_flagged() -> None:
    gold = make_target(make_card(weight_total=kg(12592)))
    tonnes = make_target(make_card(weight_total=Quantity(value=12592, unit="t")))
    pounds = make_target(make_card(weight_total=Quantity(value=12592, unit="lb")))
    other = make_target(make_card(weight_total=kg(9000)))
    assert "wrong_unit_magnitude" in score(gold, tonnes).critical_errors
    assert "wrong_unit_magnitude" in score(gold, pounds).critical_errors
    assert "wrong_unit_magnitude" not in score(gold, other).critical_errors
    assert score(gold, other).field_correct["weight_total"] is False


def test_od_swap_flagged() -> None:
    gold = make_target()
    card = gold.card
    swapped = make_target(make_card(origin=card.destination, destination=card.origin))
    case = score(gold, swapped)
    assert "origin_destination_swap" in case.critical_errors
    assert case.key_fields_correct is False
    # a single wrong city is an error, but not a swap
    moved = make_target(make_card(origin=card.destination))
    assert "origin_destination_swap" not in score(gold, moved).critical_errors


def test_hallucinated_field_flagged() -> None:
    gold = make_target(make_card(pieces=None))  # the text does not state pieces
    assert gold.missing_fields == ["pieces"]
    invented = make_target(make_card(pieces=1), missing_fields=[])
    case = score(gold, invented)
    assert case.hallucinated_fields == ["pieces"]
    assert "hallucinated_required_field" in case.critical_errors
    assert case.field_correct["pieces"] is False
    assert not case.missing_exact and case.missing_fn == 1
    assert aggregate("hallucination_rate", [case]) == 1.0


def test_hallucinated_optional_field_is_not_critical() -> None:
    gold = make_target()  # dry van: temperature_c is not required
    answer = make_target(make_card(temperature_c=-18))
    case = score(gold, answer)
    assert case.hallucinated_fields == ["temperature_c"]
    assert case.critical_errors == []


def test_reefer_temperature_invented_is_critical() -> None:
    gold = make_target(make_card(equipment_type="reefer"))
    assert "temperature_c" in gold.missing_fields
    answer = make_target(make_card(equipment_type="reefer", temperature_c=-18),
                         missing_fields=[])  # fmt: skip
    assert "hallucinated_required_field" in score(gold, answer).critical_errors


def test_code_fence_fails_strict_passes_lenient() -> None:
    gold = make_target()
    case = score(gold, f"```json\n{answer_of(gold)}\n```")
    assert not case.json_parsed and not case.schema_valid
    assert case.json_parsed_lenient and not case.failed


def test_code_fence_fields_scored_by_lenient_parse() -> None:
    gold = make_target()
    case = score(gold, f"```json\n{answer_of(gold)}\n```")
    assert aggregate("json_valid_rate", [case]) == 0.0
    assert all(v is True for v in case.field_correct.values() if v is not None)
    assert case.key_fields_correct is True


def test_schema_error_is_parsed_but_not_valid() -> None:
    case = score(make_target(), '{"card": {}, "missing_fields": [], "conflicts": []}')
    assert case.json_parsed and not case.schema_valid and case.failed


def test_invalid_json_all_fields_false() -> None:
    gold = make_target(make_card(pieces=None))
    case = score(gold, "not json at all")
    assert not case.json_parsed and not case.json_parsed_lenient and case.failed
    for field in FIELDS:
        expected = None if getattr(gold.card, field) is None else False
        assert case.field_correct[field] is expected
    assert case.key_fields_correct is None  # pieces is null in the gold
    assert case.missing_fn == 1 and not case.missing_exact
    # a failed answer is not assessable for hallucinations and critical errors
    assert aggregate("hallucination_rate", [case]) is None
    assert aggregate("critical_error_count", [case]) is None
    assert aggregate("failed_output_rate", [case]) == 1.0


def test_generation_error_is_a_failed_case() -> None:
    gold = make_target()
    case = score(gold, answer_of(gold), generation_error="timeout")
    assert case.failed and case.generation_error == "timeout"
    assert case.key_fields_correct is False
    assert aggregate("key_field_accuracy", [case]) == 0.0


def test_missing_f1_math() -> None:
    gold = make_target(make_card(pieces=None, pickup_date=None))
    assert gold.missing_fields == ["pickup_date", "pieces"]
    answer = make_target(make_card(pieces=None, pickup_date=date(2022, 1, 1)),
                         missing_fields=["pieces", "weight_total"])  # fmt: skip
    case = score(gold, answer)
    assert (case.missing_tp, case.missing_fp, case.missing_fn) == (1, 1, 1)
    perfect = score(make_target(), make_target())
    # pooled: TP 1, FP 1, FN 1 -> 2 / (2 + 1 + 1)
    assert aggregate("missing_f1", [case, perfect]) == pytest.approx(0.5)
    assert aggregate("missing_exact_rate", [case, perfect]) == 0.5
    assert aggregate("missing_f1", [perfect]) == 1.0  # nothing missing, nothing claimed


def conflict_gold() -> ExtractionTarget:
    card = make_card(weight_total=None)  # as generated: the conflict field is also missing
    return make_target(card, conflicts=[Conflict(field="weight_total",
                                                 values=[kg(12000), kg(13000)])])  # fmt: skip


def test_missed_conflict_flagged() -> None:
    gold = conflict_gold()
    confident = make_target(make_card(weight_total=kg(12000)))
    case = score(gold, confident)
    assert "missed_conflict" in case.critical_errors
    assert case.hallucinated_fields == []  # a conflict field is a missed conflict, not both
    assert case.conflict_detected is False
    assert aggregate("conflict_recall", [case]) == 0.0
    noticed = score(gold, gold)
    assert noticed.conflict_detected is True and noticed.critical_errors == []


def test_null_gold_null_pred_not_counted_as_error() -> None:
    gold = make_target()  # temperature_c and weight_per_piece are null
    case = score(gold, gold)
    assert case.field_correct["temperature_c"] is None
    assert case.field_correct["weight_per_piece"] is None
    assert aggregate("field_accuracy.temperature_c", [case]) is None
    assert aggregate("conflict_recall", [case]) is None  # no conflict in the gold


def test_computed_total_is_noted_not_hallucinated() -> None:
    card = make_card(weight_total=None, weight_per_piece=kg(500), pieces=4)
    gold = make_target(card)
    multiplied = make_target(make_card(weight_total=kg(2000), weight_per_piece=kg(500), pieces=4))
    case = score(gold, multiplied)
    assert case.notes == ["computed_weight_total"]
    assert case.hallucinated_fields == [] and case.critical_errors == []
    assert case.field_correct["weight_total"] is False
    assert case.key_fields_correct is True  # the stated mass (2000 kg) is right


def test_key_fields_need_the_stated_mass() -> None:
    gold = make_target()
    no_weight = make_target(make_card(weight_total=None))
    assert score(gold, no_weight).key_fields_correct is False
    # the gold without pieces has no stated mass: key accuracy does not apply
    per_piece = make_target(make_card(pieces=None, weight_total=None, weight_per_piece=kg(10)))
    assert card_mass_kg(per_piece.card) is None
    assert score(per_piece, per_piece).key_fields_correct is None


def test_values_match_places_and_shipper() -> None:
    place = Place(city="Пермь", region="Пермский край")
    assert values_match("origin", place, Place(city=" пермь ", region="ПЕРМСКИЙ КРАЙ"))
    assert not values_match("origin", place, Place(city="Пермь", region="Самарская область"))
    assert values_match("shipper_name", "National  Retail", "national retail")
    assert values_match("pickup_date", date(2022, 1, 1), date(2022, 1, 1))
    assert not values_match("pieces", 22, None)
    assert values_match("pieces", None, None)


def test_critical_error_counts_by_type() -> None:
    gold = make_target()
    card = gold.card
    swapped = score(gold, make_target(make_card(origin=card.destination,
                                                destination=card.origin)))  # fmt: skip
    for kind in CRITICAL_ERRORS:
        expected = 1.0 if kind == "origin_destination_swap" else 0.0
        assert aggregate(f"critical_error_count.{kind}", [swapped]) == expected
    assert aggregate("critical_error_count", [swapped, swapped]) == 2.0


def test_field_accuracy_micro_pools_fields() -> None:
    gold = make_target()
    wrong_date = score(gold, make_target(make_card(pickup_date=date(2022, 1, 5))))
    scored = sum(v is not None for v in wrong_date.field_correct.values())
    assert aggregate("field_accuracy_micro", [wrong_date]) == pytest.approx((scored - 1) / scored)
    assert aggregate("field_accuracy.pickup_date", [wrong_date]) == 0.0
    assert aggregate("field_accuracy_micro", []) is None
