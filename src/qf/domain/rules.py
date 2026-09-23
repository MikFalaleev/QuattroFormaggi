"""The missing-fields rule and consistency of an answer (plan B.3, D-042).

`missing_fields` is computed only by `compute_missing_fields`, never by an LLM or a person.
"""

from __future__ import annotations

from typing import Final

from qf.contracts import ExtractionTarget, FieldName, ShipmentCard
from qf.domain.units import to_kg

__all__ = [
    "REQUIRED_ORDER",
    "check_target_consistency",
    "compute_missing_fields",
    "total_weight_kg",
]

REQUIRED_ORDER: Final[tuple[FieldName, ...]] = (
    "origin",
    "destination",
    "pickup_date",
    "equipment_type",
    "pieces",
    "weight_total",
    "temperature_c",
)


def compute_missing_fields(card: ShipmentCard) -> list[FieldName]:
    """Required fields that are null, in `REQUIRED_ORDER`.

    - origin, destination, pickup_date, equipment_type, pieces are always required;
    - weight_total is missing only if neither weight_total nor weight_per_piece is given
      (per-piece weight without pieces reports only `pieces`);
    - temperature_c is required only for a reefer;
    - delivery_date, shipper_name, cargo_category are optional.
    """
    absent: dict[FieldName, bool] = {
        "origin": card.origin is None,
        "destination": card.destination is None,
        "pickup_date": card.pickup_date is None,
        "equipment_type": card.equipment_type is None,
        "pieces": card.pieces is None,
        "weight_total": card.weight_total is None and card.weight_per_piece is None,
        "temperature_c": card.equipment_type == "reefer" and card.temperature_c is None,
    }
    return [name for name in REQUIRED_ORDER if absent[name]]


def total_weight_kg(card: ShipmentCard) -> float | None:
    """Total mass in kg: weight_total, else weight_per_piece x pieces, else None."""
    if card.weight_total is not None:
        return to_kg(card.weight_total)
    if card.weight_per_piece is not None and card.pieces is not None:
        return to_kg(card.weight_per_piece) * card.pieces
    return None


def check_target_consistency(target: ExtractionTarget) -> list[str]:
    """Violations of the card_v1 rules; an empty list means the answer is consistent."""
    problems = []
    expected = compute_missing_fields(target.card)
    if target.missing_fields != expected:
        problems.append(f"missing_fields {target.missing_fields} != rule {expected}")
    seen: set[FieldName] = set()
    for conflict in target.conflicts:
        if conflict.field in seen:
            problems.append(f"conflict for '{conflict.field}' is listed more than once")
        seen.add(conflict.field)
        if getattr(target.card, conflict.field) is not None:
            problems.append(f"conflicting field '{conflict.field}' must be null in the card")
    if target.card.weight_total is not None and target.card.weight_per_piece is not None:
        problems.append("weight_total and weight_per_piece are both set; card_v1 uses one")
    return problems
