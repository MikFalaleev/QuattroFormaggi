"""The missing-fields rule and consistency of a `card_v2` answer (D-084).

`missing_fields` is computed only by `compute_missing_fields_v2`, never by an LLM or a person.
A condition that the request does not name is absent and not asked for, unless the equipment
type needs it (`REQUIRED_CONDITIONS`); a condition named without its values is asked for.
"""

from __future__ import annotations

from typing import Final, get_args

from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    ConditionKind,
    EquipmentTypeV2,
    ExtractionTargetV2,
    MissingFieldV2,
    OversizeCondition,
    PackagingCondition,
    PackagingType,
    SecuringCondition,
    SecuringMethod,
    SensorParameter,
    SensorsCondition,
    ShipmentCardV2,
    SpecialCondition,
    TemperatureCondition,
)

__all__ = [
    "BASE_REQUIRED_ORDER_V2",
    "CONDITION_MISSING_NAMES",
    "REQUIRED_CONDITIONS",
    "check_target_consistency_v2",
    "compute_missing_fields_v2",
    "condition_lacks_values",
    "has_special_conditions",
]

BASE_REQUIRED_ORDER_V2: Final[tuple[MissingFieldV2, ...]] = (
    "origin", "destination", "pickup_date", "equipment_type", "pieces", "weight_total",
)  # fmt: skip
REQUIRED_CONDITIONS: Final[dict[EquipmentTypeV2, tuple[ConditionKind, ...]]] = {
    "reefer": ("temperature",),
    "lowbed": ("oversize",),
}
"""Conditions without which the equipment cannot be dispatched (D-084): a reefer needs the
temperature regime, a lowbed trailer all three dimensions of the cargo."""
CONDITION_MISSING_NAMES: Final[dict[ConditionKind, MissingFieldV2]] = {
    "temperature": "special_conditions.temperature",
    "securing": "special_conditions.securing",
    "packaging": "special_conditions.packaging",
    "oversize": "special_conditions.oversize",
    "sensors": "special_conditions.sensors",
}
"""The name of a condition in `missing_fields`."""
_LIST_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "methods": get_args(SecuringMethod),
    "types": get_args(PackagingType),
    "parameters": get_args(SensorParameter),
}


def condition_lacks_values(condition: SpecialCondition, *, required: bool = False) -> bool:
    """The condition is named but its values are not known: nothing to act on. A required
    oversize condition needs all three dimensions, any other one needs at least one value."""
    match condition:
        case TemperatureCondition():
            return condition.min_c is None and condition.max_c is None
        case OversizeCondition():
            dims = (condition.length, condition.width, condition.height)
            return any(d is None for d in dims) if required else all(d is None for d in dims)
        case SecuringCondition():
            return not condition.methods
        case PackagingCondition():
            return not condition.types
        case SensorsCondition():
            return not condition.parameters


def has_special_conditions(card: ShipmentCardV2) -> bool:
    """«Особые условия: да/нет» (D-081): computed by code, never written by the model."""
    return bool(card.special_conditions)


def compute_missing_fields_v2(card: ShipmentCardV2) -> list[MissingFieldV2]:
    """Required fields that are null (as in card_v1, without temperature_c), then the
    conditions that are needed but absent or without values, in `CONDITION_KINDS` order."""
    absent: dict[MissingFieldV2, bool] = {
        "origin": card.origin is None,
        "destination": card.destination is None,
        "pickup_date": card.pickup_date is None,
        "equipment_type": card.equipment_type is None,
        "pieces": card.pieces is None,
        "weight_total": card.weight_total is None and card.weight_per_piece is None,
    }
    missing = [name for name in BASE_REQUIRED_ORDER_V2 if absent[name]]
    needed = REQUIRED_CONDITIONS.get(card.equipment_type, ()) if card.equipment_type else ()
    stated = {condition.kind: condition for condition in card.special_conditions}
    for kind in CARD_V2_CONDITION_KINDS:
        condition = stated.get(kind)
        required = kind in needed
        if (condition is None and required) or (
            condition is not None and condition_lacks_values(condition, required=required)
        ):
            missing.append(CONDITION_MISSING_NAMES[kind])
    return missing


def check_target_consistency_v2(target: ExtractionTargetV2) -> list[str]:
    """Violations of the card_v2 rules, including the canonical order of the conditions and
    of their lists; an empty list means the answer is consistent."""
    problems = []
    card = target.card
    expected = compute_missing_fields_v2(card)
    if target.missing_fields != expected:
        problems.append(f"missing_fields {target.missing_fields} != rule {expected}")
    seen: set[str] = set()
    for conflict in target.conflicts:
        if conflict.field in seen:
            problems.append(f"conflict for '{conflict.field}' is listed more than once")
        seen.add(conflict.field)
        if getattr(card, conflict.field) is not None:
            problems.append(f"conflicting field '{conflict.field}' must be null in the card")
    if card.weight_total is not None and card.weight_per_piece is not None:
        problems.append("weight_total and weight_per_piece are both set; card_v2 uses one")
    kinds = [condition.kind for condition in card.special_conditions]
    if kinds != sorted(kinds, key=CARD_V2_CONDITION_KINDS.index):
        problems.append(
            f"special_conditions {kinds} are not in the order {CARD_V2_CONDITION_KINDS}"
        )
    for condition in card.special_conditions:
        for name, order in _LIST_ORDER.items():
            values: list[str] | None = getattr(condition, name, None)
            if values is not None and values != sorted(values, key=order.index):
                problems.append(f"{condition.kind}.{name} {values} are not in the order {order}")
    return problems
