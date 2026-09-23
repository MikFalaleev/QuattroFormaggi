from __future__ import annotations

import pytest

from qf.contracts import Conflict, Quantity, ShipmentCard
from qf.domain import (
    REQUIRED_ORDER,
    check_target_consistency,
    compute_missing_fields,
    total_weight_kg,
)
from tests.factories import make_card, make_target


def test_full_dry_van_has_no_missing() -> None:
    assert compute_missing_fields(make_card()) == []


def test_reefer_requires_temperature() -> None:
    assert compute_missing_fields(make_card(equipment_type="reefer")) == ["temperature_c"]
    assert compute_missing_fields(make_card(equipment_type="reefer", temperature_c=-18.0)) == []


def test_unknown_equipment_does_not_require_temperature() -> None:
    assert compute_missing_fields(make_card(equipment_type=None)) == ["equipment_type"]


def test_per_piece_with_pieces_counts_as_weight_known() -> None:
    card = make_card(weight_total=None, weight_per_piece=Quantity(value=572, unit="kg"))
    assert compute_missing_fields(card) == []


def test_per_piece_without_pieces_missing_pieces_only() -> None:
    card = make_card(
        weight_total=None, weight_per_piece=Quantity(value=572, unit="kg"), pieces=None
    )
    assert compute_missing_fields(card) == ["pieces"]


def test_no_weight_at_all_reports_weight_total() -> None:
    assert compute_missing_fields(make_card(weight_total=None)) == ["weight_total"]


def test_optional_fields_are_never_missing() -> None:
    card = make_card(shipper_name=None, cargo_category=None, delivery_date=None)
    assert compute_missing_fields(card) == []


def test_missing_order_is_canonical() -> None:
    empty = make_card(**dict.fromkeys(ShipmentCard.model_fields))
    assert compute_missing_fields(empty) == [
        "origin", "destination", "pickup_date", "equipment_type", "pieces", "weight_total",
    ]  # fmt: skip
    reefer = empty.model_copy(update={"equipment_type": "reefer"})
    assert compute_missing_fields(reefer) == [f for f in REQUIRED_ORDER if f != "equipment_type"]


def test_total_weight_kg() -> None:
    assert total_weight_kg(make_card()) == pytest.approx(12592.18778357)
    per_piece = make_card(weight_total=None, weight_per_piece=Quantity(value=572, unit="kg"))
    assert total_weight_kg(per_piece) == 572 * 22
    assert total_weight_kg(per_piece.model_copy(update={"pieces": None})) is None
    assert total_weight_kg(make_card(weight_total=None)) is None


def test_consistent_target_has_no_problems() -> None:
    assert check_target_consistency(make_target()) == []


def test_consistency_detects_wrong_missing_list() -> None:
    card = make_card(origin=None, equipment_type="reefer")
    assert check_target_consistency(make_target(card)) == []
    for wrong in ([], ["origin"], ["temperature_c", "origin"], ["origin", "origin"]):
        problems = check_target_consistency(make_target(card, missing_fields=wrong))
        assert len(problems) == 1
        assert "missing_fields" in problems[0]


def test_conflict_field_must_be_null() -> None:
    conflict = Conflict(field="pieces", values=[18, 20])
    problems = check_target_consistency(make_target(conflicts=[conflict]))
    assert problems == ["conflicting field 'pieces' must be null in the card"]
    card = make_card(pieces=None)
    assert check_target_consistency(make_target(card, conflicts=[conflict])) == []


def test_conflict_listed_once_per_field() -> None:
    card = make_card(pieces=None)
    conflicts = [Conflict(field="pieces", values=[18, 20]), Conflict(field="pieces", values=[1, 2])]
    problems = check_target_consistency(make_target(card, conflicts=conflicts))
    assert problems == ["conflict for 'pieces' is listed more than once"]


def test_both_weights_set_is_inconsistent() -> None:
    card = make_card(weight_per_piece=Quantity(value=572, unit="kg"))
    assert check_target_consistency(make_target(card)) == [
        "weight_total and weight_per_piece are both set; card_v1 uses one"
    ]
