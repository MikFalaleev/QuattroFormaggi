"""Scoring of one model answer against the gold (plan step 9, docs/EVAL_SPEC.md).

The answer is parsed once, strictly (for json_valid_rate) and leniently (for everything else);
metrics only read the resulting `CaseScore`. A case whose answer cannot be scored (generation
error, or not even the lenient parse) is counted as failed, never skipped.
"""

from __future__ import annotations

from typing import Any, Final, cast, get_args

from qf.contracts import CaseScore, ExtractionTarget, FieldName, Place, Quantity, ShipmentCard
from qf.domain import TargetParseError, parse_target, parse_target_lenient, to_kg

__all__ = [
    "CRITICAL_ERRORS",
    "FIELDS",
    "KEY_FIELDS",
    "WEIGHT_TOLERANCE_KG",
    "card_mass_kg",
    "score_prediction",
    "values_match",
]

FIELDS: Final = cast(tuple[FieldName, ...], get_args(FieldName))  # card field order
# Key fields of key_field_accuracy, together with the stated mass (card_mass_kg).
KEY_FIELDS: Final[tuple[FieldName, ...]] = (
    "origin", "destination", "pickup_date", "equipment_type", "pieces",
)  # fmt: skip
WEIGHT_TOLERANCE_KG: Final = 0.5
CRITICAL_ERRORS: Final = (
    "pieces_weight_swap",
    "wrong_unit_magnitude",
    "origin_destination_swap",
    "hallucinated_required_field",
    "missed_conflict",
)
_UNIT_RATIOS: Final = (0.45359237, 1 / 0.45359237, 1000.0, 0.001)
_RATIO_TOLERANCE: Final = 0.02


def _same_place(a: Place, b: Place) -> bool:
    return (a.city.strip().casefold(), a.region.strip().casefold()) == (
        b.city.strip().casefold(), b.region.strip().casefold(),
    )  # fmt: skip


def values_match(field: FieldName, gold: Any, answer: Any) -> bool:
    """Weights by kg within 0.5 kg (the unit itself is not compared); places by city and
    region ignoring case and outer spaces; the shipper ignoring case and extra spaces; the
    rest exactly."""
    if gold is None or answer is None:
        return gold is None and answer is None
    if field in ("weight_total", "weight_per_piece"):
        return abs(to_kg(gold) - to_kg(answer)) <= WEIGHT_TOLERANCE_KG
    if field in ("origin", "destination"):
        return _same_place(gold, answer)
    if field == "shipper_name":
        return " ".join(gold.casefold().split()) == " ".join(answer.casefold().split())
    return bool(gold == answer)


def card_mass_kg(card: ShipmentCard) -> float | None:
    """Total mass stated by the card: weight_total, or weight_per_piece x pieces when both are
    given. Unlike `total_weight_kg` there is no default of one piece (D-050): a card without
    pieces has no stated mass."""
    if card.weight_total is not None:
        return to_kg(card.weight_total)
    if card.weight_per_piece is not None and card.pieces is not None:
        return to_kg(card.weight_per_piece) * card.pieces
    return None


def _parse(raw: str) -> tuple[bool, bool, ExtractionTarget | None]:
    """(strict format ok, strictly valid, leniently parsed answer or None)."""
    try:
        parse_target(raw)
        json_parsed, schema_valid = True, True
    except TargetParseError as exc:
        json_parsed, schema_valid = exc.stage == "schema", False
    try:
        return json_parsed, schema_valid, parse_target_lenient(raw)
    except TargetParseError:
        return json_parsed, schema_valid, None


def _failed(record_id: str, gold: ExtractionTarget, flags: tuple[bool, bool],
            generation_error: str | None) -> CaseScore:  # fmt: skip
    card = gold.card
    key_applies = card_mass_kg(card) is not None and all(
        getattr(card, f) is not None for f in KEY_FIELDS
    )
    return CaseScore(
        record_id=record_id, json_parsed=flags[0], schema_valid=flags[1],
        json_parsed_lenient=False, generation_error=generation_error,
        field_correct={f: (False if getattr(card, f) is not None else None) for f in FIELDS},
        key_fields_correct=False if key_applies else None,
        missing_tp=0, missing_fp=0, missing_fn=len(gold.missing_fields), missing_exact=False,
        conflict_detected=False if gold.conflicts else None,
        hallucinated_fields=[], critical_errors=[], notes=[],
    )  # fmt: skip


def _numbers(quantity: Quantity | None) -> set[float]:
    return set() if quantity is None else {float(quantity.value)}


def _critical(gold: ShipmentCard, answer: ShipmentCard) -> list[str]:
    errors = []
    gold_weights = _numbers(gold.weight_total) | _numbers(gold.weight_per_piece)
    answer_weights = _numbers(answer.weight_total) | _numbers(answer.weight_per_piece)
    swapped_pieces = answer.pieces is not None and answer.pieces != gold.pieces and (
        float(answer.pieces) in gold_weights)  # fmt: skip
    swapped_weight = gold.pieces is not None and float(gold.pieces) in answer_weights and (
        float(gold.pieces) not in gold_weights)  # fmt: skip
    if swapped_pieces or swapped_weight:
        errors.append("pieces_weight_swap")
    for field in ("weight_total", "weight_per_piece"):
        g, a = getattr(gold, field), getattr(answer, field)
        if g is not None and a is not None and not values_match(field, g, a) and to_kg(g) > 0:
            ratio = to_kg(a) / to_kg(g)
            if any(abs(ratio / r - 1) <= _RATIO_TOLERANCE for r in _UNIT_RATIOS):
                errors.append("wrong_unit_magnitude")
                break
    if (gold.origin and gold.destination and answer.origin and answer.destination
            and not _same_place(gold.origin, gold.destination)
            and _same_place(answer.origin, gold.destination)
            and _same_place(answer.destination, gold.origin)):  # fmt: skip
        errors.append("origin_destination_swap")
    return errors


def score_prediction(
    record_id: str, gold: ExtractionTarget, raw_output: str, *, generation_error: str | None = None
) -> CaseScore:
    """Score one answer. Fields are compared on the lenient parse; `missing_*` against the gold
    list as sets (never recomputed from the answer's card)."""
    json_parsed, schema_valid, answer = _parse(raw_output)
    if generation_error is not None or answer is None:
        return _failed(record_id, gold, (json_parsed, schema_valid), generation_error)
    g, a = gold.card, answer.card
    conflict_fields = {c.field for c in gold.conflicts}
    field_correct: dict[FieldName, bool | None] = {}
    hallucinated: list[FieldName] = []
    critical: list[str] = []
    notes: list[str] = []
    for field in FIELDS:
        gold_value, answer_value = getattr(g, field), getattr(a, field)
        if gold_value is None and answer_value is None:
            field_correct[field] = None
        elif gold_value is None:
            field_correct[field] = False
            if field in conflict_fields:
                critical.append("missed_conflict")
            elif field == "weight_total" and _is_computed_total(g, answer_value):
                notes.append("computed_weight_total")
            else:
                hallucinated.append(field)
        else:
            field_correct[field] = values_match(field, gold_value, answer_value)
    # required for this card and absent from the text (the gold lists it as missing), yet the
    # answer states a value: e.g. temperature_c of a reefer, but not of a dry van
    if any(f in gold.missing_fields for f in hallucinated):
        critical.append("hallucinated_required_field")
    critical += _critical(g, a)
    gold_missing, answer_missing = set(gold.missing_fields), set(answer.missing_fields)
    detected = conflict_fields <= {c.field for c in answer.conflicts} if conflict_fields else None
    return CaseScore(
        record_id=record_id, json_parsed=json_parsed, schema_valid=schema_valid,
        json_parsed_lenient=True, generation_error=None, field_correct=field_correct,
        key_fields_correct=_key_correct(g, a, field_correct),
        missing_tp=len(gold_missing & answer_missing),
        missing_fp=len(answer_missing - gold_missing),
        missing_fn=len(gold_missing - answer_missing),
        missing_exact=gold_missing == answer_missing, conflict_detected=detected,
        hallucinated_fields=hallucinated, critical_errors=sorted(set(critical)), notes=notes,
    )  # fmt: skip


def _key_correct(
    gold: ShipmentCard, answer: ShipmentCard, field_correct: dict[FieldName, bool | None]
) -> bool | None:
    """All key fields and the stated mass right; None if any of them is null in the gold."""
    gold_mass = card_mass_kg(gold)
    if gold_mass is None or any(getattr(gold, f) is None for f in KEY_FIELDS):
        return None
    answer_mass = card_mass_kg(answer)
    mass_ok = answer_mass is not None and abs(answer_mass - gold_mass) <= WEIGHT_TOLERANCE_KG
    return mass_ok and all(field_correct[f] for f in KEY_FIELDS)


def _is_computed_total(gold: ShipmentCard, answer_total: Quantity) -> bool:
    """The answer multiplied the per-piece weight out: wrong field, but not an invented value."""
    mass = card_mass_kg(gold)
    return gold.weight_per_piece is not None and mass is not None and (
        abs(to_kg(answer_total) - mass) <= WEIGHT_TOLERANCE_KG)  # fmt: skip
