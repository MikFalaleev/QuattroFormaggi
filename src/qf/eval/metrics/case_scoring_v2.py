"""Scoring of one card_v2 answer (sub-step V6, D-085, docs/EVAL_SPEC.md).

The fields both card versions share are scored exactly as in card_v1 (`score_prediction`). The
special conditions are compared by kind: lists as sets, dimensions in metres within 0.01 m,
temperatures exactly, so the order of conditions and of list values does not matter. Every
condition the answer adds or loses gets exactly one critical label (the principle of D-075):

| gold | answer | label |
|---|---|---|
| has the kind | lacks it | `missed_condition` |
| lacks it, `special_conditions.<kind>` missing (a reefer without a temperature) | has it with a value | `hallucinated_required_field` |
| the same | has it without values | wrong condition, not critical: the answer still asks |
| lacks it, not missing | has it | `hallucinated_condition` |
| has it and lists it as missing (no values, part of the dimensions) | a value the gold lacks | `hallucinated_required_field` |
| has it | other values | wrong condition, not critical |
"""  # noqa: E501

from __future__ import annotations

from typing import Any, Final, cast, get_args

from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    AnyFieldName,
    CaseScoreV2,
    ConditionKind,
    ExtractionTargetV2,
    FieldNameV2,
    Length,
    SpecialCondition,
)
from qf.domain import parse_target_lenient_v2, parse_target_v2, to_m
from qf.eval.metrics.case_scoring import (
    CRITICAL_ERRORS,
    KEY_FIELDS,
    card_mass_kg,
    is_computed_total,
    key_correct,
    parse_answer,
    scalar_critical_errors,
    values_match,
)

__all__ = [
    "CONDITIONS_FIELD",
    "CRITICAL_ERRORS_V2",
    "FIELDS_V2",
    "LENGTH_TOLERANCE_M",
    "conditions_match",
    "score_prediction_v2",
]

FIELDS_V2: Final = cast(tuple[FieldNameV2, ...], get_args(FieldNameV2))  # card field order
CONDITIONS_FIELD: Final = "special_conditions"
_SCALAR_FIELDS: Final = tuple(f for f in FIELDS_V2 if f != CONDITIONS_FIELD)
CRITICAL_ERRORS_V2: Final = (*CRITICAL_ERRORS, "hallucinated_condition", "missed_condition")
LENGTH_TOLERANCE_M: Final = 0.01


def _value_names(condition: SpecialCondition) -> list[str]:
    return [name for name in type(condition).model_fields if name != "kind"]


def _same_value(gold: Any, answer: Any) -> bool:
    if isinstance(gold, list) or isinstance(answer, list):
        return set(gold or ()) == set(answer or ())
    if gold is None or answer is None:
        return gold is None and answer is None
    if isinstance(gold, Length) and isinstance(answer, Length):
        return abs(to_m(gold) - to_m(answer)) <= LENGTH_TOLERANCE_M
    return bool(gold == answer)


def conditions_match(gold: SpecialCondition, answer: SpecialCondition) -> bool:
    """Same kind and the same values: lists as sets, dimensions in metres within 0.01 m."""
    return gold.kind == answer.kind and all(
        _same_value(getattr(gold, name), getattr(answer, name)) for name in _value_names(gold)
    )


def _invents(gold: SpecialCondition | None, answer: SpecialCondition) -> bool:
    """The answer states a value the gold lacks: a list value not in the gold list, a number or
    a dimension where the gold has null (or has no such condition at all)."""
    for name in _value_names(answer):
        value = getattr(answer, name)
        known = getattr(gold, name) if gold is not None else None
        if isinstance(value, list):
            if set(value) - set(known or ()):
                return True
        elif value is not None and known is None:
            return True
    return False


def _failed(record_id: str, gold: ExtractionTargetV2, flags: tuple[bool, bool],
            generation_error: str | None) -> CaseScoreV2:  # fmt: skip
    card = gold.card
    kinds = [c.kind for c in card.special_conditions]
    field_correct: dict[AnyFieldName, bool | None] = {
        f: (False if getattr(card, f) is not None else None) for f in _SCALAR_FIELDS
    }
    field_correct[CONDITIONS_FIELD] = False if kinds else None
    key_applies = card_mass_kg(card) is not None and all(
        getattr(card, f) is not None for f in KEY_FIELDS
    )
    return CaseScoreV2(
        record_id=record_id, json_parsed=flags[0], schema_valid=flags[1],
        json_parsed_lenient=False, generation_error=generation_error, field_correct=field_correct,
        key_fields_correct=False if key_applies else None,
        missing_tp=0, missing_fp=0, missing_fn=len(gold.missing_fields), missing_exact=False,
        conflict_detected=False if gold.conflicts else None,
        hallucinated_fields=[], critical_errors=[], notes=[],
        conditions_gold=kinds, conditions_answer=[],
        condition_correct={k: (False if k in kinds else None) for k in CARD_V2_CONDITION_KINDS},
    )  # fmt: skip


def _score_conditions(
    gold: ExtractionTargetV2, answer: ExtractionTargetV2
) -> tuple[dict[ConditionKind, bool | None], list[str], bool]:
    """(correctness by kind, critical labels, whether the answer states anything the gold does
    not have)."""
    gold_by = {c.kind: c for c in gold.card.special_conditions}
    answer_by = {c.kind: c for c in answer.card.special_conditions}
    correct: dict[ConditionKind, bool | None] = {}
    critical: list[str] = []
    invented_any = False
    for kind in CARD_V2_CONDITION_KINDS:
        known, stated = gold_by.get(kind), answer_by.get(kind)
        asked = f"{CONDITIONS_FIELD}.{kind}" in gold.missing_fields
        if known is None and stated is None:
            correct[kind] = None
            continue
        if stated is None:
            correct[kind] = False
            critical.append("missed_condition")
            continue
        invents = _invents(known, stated)
        invented_any = invented_any or invents or known is None
        correct[kind] = known is not None and conditions_match(known, stated)
        if asked and invents:
            critical.append("hallucinated_required_field")
        elif known is None and not asked:
            critical.append("hallucinated_condition")
    return correct, critical, invented_any


def score_prediction_v2(
    record_id: str, gold: ExtractionTargetV2, raw_output: str, *,
    generation_error: str | None = None,
) -> CaseScoreV2:  # fmt: skip
    """Score one card_v2 answer. Shared fields as in `score_prediction`; conditions by kind
    (module docstring); `missing_*` against the gold list as sets. Special conditions are a
    key field: `key_fields_correct` also needs all of them right."""
    json_parsed, schema_valid, answer = parse_answer(raw_output, parse_target_v2,
                                                     parse_target_lenient_v2)  # fmt: skip
    if generation_error is not None or answer is None:
        return _failed(record_id, gold, (json_parsed, schema_valid), generation_error)
    g, a = gold.card, answer.card
    conflict_fields = {c.field for c in gold.conflicts}
    field_correct: dict[AnyFieldName, bool | None] = {}
    hallucinated: list[AnyFieldName] = []
    critical: list[str] = []
    notes: list[str] = []
    for field in _SCALAR_FIELDS:
        gold_value, answer_value = getattr(g, field), getattr(a, field)
        if gold_value is None and answer_value is None:
            field_correct[field] = None
        elif gold_value is None:
            field_correct[field] = False
            if field in conflict_fields:
                critical.append("missed_conflict")
            elif field == "weight_total" and is_computed_total(g, answer_value):
                notes.append("computed_weight_total")
            else:
                hallucinated.append(field)
        else:
            field_correct[field] = values_match(field, gold_value, answer_value)
    if any(f in gold.missing_fields for f in hallucinated):
        critical.append("hallucinated_required_field")
    critical += scalar_critical_errors(g, a)
    by_kind, condition_errors, invented = _score_conditions(gold, answer)
    critical += condition_errors
    if g.special_conditions or a.special_conditions:
        field_correct[CONDITIONS_FIELD] = all(v is not False for v in by_kind.values())
    else:
        field_correct[CONDITIONS_FIELD] = None
    if invented:
        hallucinated.append(CONDITIONS_FIELD)
    key = key_correct(g, a, field_correct)  # None when a key field of the gold is null
    if key is not None:
        key = key and field_correct[CONDITIONS_FIELD] is not False
    gold_missing, answer_missing = set(gold.missing_fields), set(answer.missing_fields)
    detected = conflict_fields <= {c.field for c in answer.conflicts} if conflict_fields else None
    answer_kinds = {c.kind for c in a.special_conditions}
    return CaseScoreV2(
        record_id=record_id, json_parsed=json_parsed, schema_valid=schema_valid,
        json_parsed_lenient=True, generation_error=None, field_correct=field_correct,
        key_fields_correct=key,
        missing_tp=len(gold_missing & answer_missing),
        missing_fp=len(answer_missing - gold_missing),
        missing_fn=len(gold_missing - answer_missing),
        missing_exact=gold_missing == answer_missing, conflict_detected=detected,
        hallucinated_fields=hallucinated, critical_errors=sorted(set(critical)), notes=notes,
        conditions_gold=[c.kind for c in g.special_conditions],
        conditions_answer=[k for k in CARD_V2_CONDITION_KINDS if k in answer_kinds],
        condition_correct=by_kind,
    )  # fmt: skip
