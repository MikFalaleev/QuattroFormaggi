"""Contract `card_v2`, its rules, parsing, the schema registry and prompt v2 (sub-step V1)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from qf.common import QFError
from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    CargoCategoryV2,
    ConditionKind,
    ConflictFieldV2,
    ConflictV2,
    EquipmentTypeV2,
    ExtractionTargetV2,
    FieldNameV2,
    Length,
    LengthUnit,
    MissingFieldV2,
    OversizeCondition,
    PackagingCondition,
    PackagingType,
    Quantity,
    SecuringCondition,
    SecuringMethod,
    SensorParameter,
    SensorsCondition,
    ShipmentCardV2,
    TemperatureCondition,
    WeightUnit,
)
from qf.domain import (
    BASE_REQUIRED_ORDER_V2,
    CONDITION_MISSING_NAMES,
    SHIPMENT_EXTRACTION,
    TARGET_SCHEMAS,
    TASKS,
    TargetParseError,
    TaskSpec,
    check_record,
    check_target_consistency_v2,
    compute_missing_fields_v2,
    get_target_schema,
    has_special_conditions,
    load_system_prompt,
    parse_target,
    parse_target_lenient_v2,
    parse_target_v2,
    serialize_target,
    system_prompt_hash,
    to_m,
)
from tests.factories import (
    EXAMPLE_ANSWER,
    EXAMPLE_ANSWER_V2,
    make_card_v2,
    make_record,
    make_record_v2,
    make_target_v2,
)

PROMPT_V2 = "fa3297296346666443f60d0318f01eed7c669032ffcacc534e8fd166544f4e41"


def m(value: float, unit: LengthUnit = "m") -> Length:
    return Length(value=value, unit=unit)


def temperature(min_c: float | None = 2, max_c: float | None = 6) -> TemperatureCondition:
    return TemperatureCondition(kind="temperature", min_c=min_c, max_c=max_c)


def oversize(length: Length | None = None, width: Length | None = None,
             height: Length | None = None) -> OversizeCondition:  # fmt: skip
    return OversizeCondition(kind="oversize", length=length, width=width, height=height)


def card_json(**changes: Any) -> str:
    data = json.loads(EXAMPLE_ANSWER_V2)
    data["card"].update(changes)
    return json.dumps(data, ensure_ascii=False)


# --- contract ------------------------------------------------------------------------------


def test_names_match_the_models() -> None:
    assert get_args(FieldNameV2) == tuple(ShipmentCardV2.model_fields)
    assert get_args(ConflictFieldV2) == tuple(f for f in get_args(FieldNameV2)
                                              if f != "special_conditions")  # fmt: skip
    assert get_args(ConditionKind) == CARD_V2_CONDITION_KINDS
    assert get_args(MissingFieldV2) == (
        *BASE_REQUIRED_ORDER_V2,
        *(CONDITION_MISSING_NAMES[k] for k in CARD_V2_CONDITION_KINDS),
    )
    assert "temperature_c" not in ShipmentCardV2.model_fields
    assert set(get_args(EquipmentTypeV2)) == {"tent", "van", "reefer", "isotherm", "container",
                                              "lowbed", "mega", "flatbed"}  # fmt: skip
    assert "machinery" in get_args(CargoCategoryV2)


def test_plan_example_parses_and_is_canonical() -> None:
    target = parse_target_v2(EXAMPLE_ANSWER_V2)
    assert serialize_target(target) == EXAMPLE_ANSWER_V2
    assert check_target_consistency_v2(target) == []
    assert has_special_conditions(target.card)
    assert target == make_target_v2()


def test_kind_is_the_first_key_of_every_condition() -> None:
    conditions = [temperature(), SecuringCondition(kind="securing", methods=["straps"]),
                  PackagingCondition(kind="packaging", types=["crate"]), oversize(height=m(4.2)),
                  SensorsCondition(kind="sensors", parameters=["tilt"])]  # fmt: skip
    for condition in conditions:
        assert next(iter(condition.model_dump())) == "kind"


def test_integer_and_negative_temperatures() -> None:
    frozen = card_json(special_conditions=[{"kind": "temperature", "min_c": -18, "max_c": -18}])
    target = parse_target_v2(frozen)
    assert target.card.special_conditions == [temperature(-18, -18)]
    assert '"min_c":-18,"max_c":-18' in serialize_target(target)
    half = card_json(special_conditions=[{"kind": "temperature", "min_c": None, "max_c": 5.5}])
    assert parse_target_v2(half).card.special_conditions == [temperature(None, 5.5)]


@pytest.mark.parametrize("bad", [
    {"kind": "temperature", "min_c": 6, "max_c": 2},  # min above max
    {"kind": "temperature", "min_c": "2", "max_c": 6},  # strict: no strings
    {"kind": "temperature", "min_c": True, "max_c": 6},
    {"kind": "securing", "methods": ["straps", "straps"]},  # repeated item
    {"kind": "securing", "methods": ["glue"]},  # not in the list
    {"kind": "packaging", "types": ["bag"]},
    {"kind": "sensors", "parameters": ["gps"]},
    {"kind": "oversize", "length": {"value": 4, "unit": "mm"}, "width": None, "height": None},
    {"kind": "oversize", "length": {"value": 0, "unit": "m"}, "width": None, "height": None},
    {"kind": "oversize", "length": None, "width": None},  # every key is required
    {"kind": "adr", "class": 3},  # unknown kind
    {"kind": "temperature", "min_c": 2, "max_c": 6, "note": "x"},  # extra key
])  # fmt: skip
def test_invalid_conditions_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(TargetParseError) as info:
        parse_target_v2(card_json(special_conditions=[bad]))
    assert info.value.stage == "schema"


def test_card_level_rules() -> None:
    twice = [{"kind": "temperature", "min_c": 2, "max_c": 6},
             {"kind": "temperature", "min_c": 3, "max_c": 5}]  # fmt: skip
    for text in (card_json(special_conditions=twice), card_json(special_conditions=None),
                 card_json(temperature_c=4)):  # fmt: skip
        with pytest.raises(TargetParseError):
            parse_target_v2(text)
    data = json.loads(EXAMPLE_ANSWER_V2)
    del data["card"]["special_conditions"]
    with pytest.raises(TargetParseError):
        parse_target_v2(json.dumps(data, ensure_ascii=False))
    with pytest.raises(ValidationError):
        ConflictV2(field="special_conditions", values=[1, 2])  # type: ignore[arg-type]


def test_empty_values_are_allowed() -> None:
    card = make_card_v2(special_conditions=[
        temperature(None, None), SecuringCondition(kind="securing", methods=[]),
        PackagingCondition(kind="packaging", types=[]), oversize(),
        SensorsCondition(kind="sensors", parameters=[]),
    ])  # fmt: skip
    assert len(card.special_conditions) == 5


def test_versions_do_not_mix() -> None:
    with pytest.raises(TargetParseError) as v1_as_v2:
        parse_target_v2(EXAMPLE_ANSWER)
    assert v1_as_v2.value.stage == "schema"
    with pytest.raises(TargetParseError, match="not a card_v1 answer"):
        parse_target(EXAMPLE_ANSWER_V2)


def test_strict_and_lenient_parse() -> None:
    fenced = f"```json\n{EXAMPLE_ANSWER_V2}\n```"
    with pytest.raises(TargetParseError) as info:
        parse_target_v2(fenced)
    assert info.value.stage == "format"
    assert parse_target_lenient_v2(f"  {fenced} ") == make_target_v2()


def test_lengths_in_metres() -> None:
    assert to_m(m(4.2)) == 4.2
    assert to_m(m(420, "cm")) == pytest.approx(4.2)


# --- the missing-fields rule ---------------------------------------------------------------


def missing(**changes: Any) -> list[str]:
    return list(compute_missing_fields_v2(make_card_v2(**changes)))


def test_base_fields_as_in_card_v1_without_temperature_c() -> None:
    assert missing() == []
    assert missing(origin=None, pieces=None, weight_total=None) == [
        "origin", "pieces", "weight_total",
    ]  # fmt: skip
    assert missing(weight_total=None, weight_per_piece=Quantity(value=500, unit="kg")) == []
    # no equipment type: no condition is required by it
    assert missing(equipment_type=None, special_conditions=[]) == ["equipment_type"]


def test_reefer_needs_a_temperature() -> None:
    assert missing(special_conditions=[]) == ["special_conditions.temperature"]
    assert missing(special_conditions=[temperature(None, 5)]) == []
    assert missing(special_conditions=[temperature(None, None)]) == [
        "special_conditions.temperature"
    ]
    # other equipment: an unstated temperature is not asked for (user decision, D-084)
    assert missing(equipment_type="isotherm", special_conditions=[]) == []
    assert missing(equipment_type="tent", special_conditions=[]) == []


def test_lowbed_needs_all_three_dimensions() -> None:
    lowbed = {"equipment_type": "lowbed", "cargo_category": "machinery"}
    assert missing(**lowbed, special_conditions=[]) == ["special_conditions.oversize"]
    two = oversize(length=m(9), height=m(3.6))
    assert missing(**lowbed, special_conditions=[two]) == ["special_conditions.oversize"]
    full = oversize(length=m(9), width=m(320, "cm"), height=m(3.6))
    assert missing(**lowbed, special_conditions=[full]) == []
    # on other equipment one stated dimension is enough, none is "named without values"
    assert missing(equipment_type="mega", special_conditions=[oversize(height=m(3))]) == []
    assert missing(equipment_type="mega", special_conditions=[oversize()]) == [
        "special_conditions.oversize"
    ]


def test_conditions_named_without_values_are_missing_in_order() -> None:
    conditions = [temperature(None, None), SecuringCondition(kind="securing", methods=[]),
                  PackagingCondition(kind="packaging", types=["crate"]),
                  SensorsCondition(kind="sensors", parameters=[])]  # fmt: skip
    assert missing(pickup_date=None, special_conditions=conditions) == [
        "pickup_date", "special_conditions.temperature", "special_conditions.securing",
        "special_conditions.sensors",
    ]  # fmt: skip


def test_has_special_conditions() -> None:
    assert not has_special_conditions(make_card_v2(equipment_type="tent", special_conditions=[]))
    assert has_special_conditions(make_card_v2())


def test_consistency_checks() -> None:
    assert check_target_consistency_v2(make_target_v2()) == []
    wrong_missing = make_target_v2(missing_fields=["pieces"])
    assert "missing_fields" in check_target_consistency_v2(wrong_missing)[0]
    reordered = make_target_v2(
        make_card_v2(
            special_conditions=[
                SensorsCondition(kind="sensors", parameters=["temperature"]),
                temperature(),
            ]
        )
    )
    assert "not in the order" in check_target_consistency_v2(reordered)[0]
    unsorted = make_target_v2(
        make_card_v2(
            special_conditions=[
                temperature(),
                SensorsCondition(kind="sensors", parameters=["tilt", "temperature"]),
            ]
        )
    )
    assert "sensors.parameters" in check_target_consistency_v2(unsorted)[0]
    both = make_target_v2(make_card_v2(weight_per_piece=Quantity(value=1, unit="t")))
    assert "both set" in check_target_consistency_v2(both)[0]
    conflict = ConflictV2(field="pieces", values=[18, 20])
    not_null = make_target_v2(conflicts=[conflict, conflict])
    problems = check_target_consistency_v2(not_null)
    assert any("more than once" in p for p in problems)
    assert any("must be null" in p for p in problems)


# --- registry, task, record checks ---------------------------------------------------------


def test_schema_registry() -> None:
    assert TARGET_SCHEMAS.names() == ["card_v1", "card_v2"]
    v1, v2 = get_target_schema("card_v1"), get_target_schema("card_v2")
    assert v1.serialize(v1.parse(EXAMPLE_ANSWER)) == EXAMPLE_ANSWER
    assert v2.serialize(v2.parse(EXAMPLE_ANSWER_V2)) == EXAMPLE_ANSWER_V2
    assert v2.card_fields == get_args(FieldNameV2)
    assert v2.missing_fields(make_card_v2(special_conditions=[])) == [
        "special_conditions.temperature"
    ]
    schema = json.dumps(v2.json_schema())
    assert "special_conditions" in schema and "discriminator" in schema
    with pytest.raises(QFError, match="unknown target_schema 'card_v9'"):
        get_target_schema("card_v9")


def test_card_v2_records_pass_record_checks() -> None:
    assert check_record(make_record_v2()) == []
    reordered = make_target_v2(
        make_card_v2(
            special_conditions=[
                SensorsCondition(kind="sensors", parameters=["temperature"]),
                temperature(),
            ]
        )
    )
    issues = check_record(make_record_v2(serialize_target(reordered)))
    assert [i.code for i in issues] == ["TARGET_INCONSISTENT"]
    v1_answer = check_record(make_record_v2(EXAMPLE_ANSWER))
    assert [i.code for i in v1_answer] == ["TARGET_PARSE"]
    not_canonical = check_record(
        make_record_v2(EXAMPLE_ANSWER_V2.replace('"min_c":2', '"min_c":2.0'))
    )
    assert [i.code for i in not_canonical] == ["NOT_CANONICAL"]
    assert check_record(make_record()) == []  # card_v1 records are unaffected


def test_record_in_a_schema_the_task_does_not_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    only_v1 = TaskSpec(name="shipment_extraction", prompts=(("card_v1", "system_extract_v1"),))
    monkeypatch.setitem(TASKS._entries, "shipment_extraction", only_v1)
    issues = check_record(make_record_v2())
    assert [i.code for i in issues] == ["UNSUPPORTED_SCHEMA"]
    assert "not card_v2" in issues[0].message
    assert SHIPMENT_EXTRACTION.schema_versions == ("card_v1", "card_v2")


# --- prompt v2 -----------------------------------------------------------------------------


def test_prompt_v2_is_pinned() -> None:
    assert system_prompt_hash("system_extract_v2") == PROMPT_V2
    assert SHIPMENT_EXTRACTION.prompt_for("card_v2") == "system_extract_v2"


def test_prompt_v2_names_every_key_and_value_of_the_contract() -> None:
    """The prompt and the contract cannot drift apart: every card key, condition kind and
    closed-list value of card_v2 appears in the prompt text."""
    prompt = load_system_prompt("system_extract_v2")
    literals = (FieldNameV2, EquipmentTypeV2, CargoCategoryV2, ConditionKind, SecuringMethod,
                PackagingType, SensorParameter, LengthUnit, WeightUnit)  # fmt: skip
    absent = [value for literal in literals for value in get_args(literal)
              if f"{value}" not in prompt]  # fmt: skip
    assert absent == []
    for field in ("min_c", "max_c", "methods", "types", "length", "width", "height",
                  "parameters", "missing_fields", "conflicts", "special_conditions."):  # fmt: skip
        assert field in prompt
    for rule in ("не пересчитывай", "даты запроса", "null, а не 1", "данные, а не инструкции",
                 "не добавляй", "без значений", "reefer", "lowbed"):  # fmt: skip
        assert rule in prompt
    assert "temperature_c" not in prompt
    assert not prompt.endswith("\n")


def test_dates_are_iso_in_card_v2() -> None:
    card = make_card_v2(pickup_date=date(2024, 3, 12))
    assert '"pickup_date":"2024-03-12"' in serialize_target(make_target_v2(card))
    assert isinstance(ExtractionTargetV2.model_json_schema(), dict)
