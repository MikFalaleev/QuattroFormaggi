from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from qf.contracts import (
    CARD_SCHEMA_VERSION,
    Conflict,
    FieldName,
    Message,
    Place,
    Quantity,
    SFTRecord,
    ShipmentCard,
    TargetSchemaVersion,
    VariantInfo,
    supported_versions,
)
from qf.domain import (
    TargetParseError,
    check_target_consistency,
    parse_target,
    parse_target_lenient,
    serialize_target,
)
from tests.conftest import REPO_ROOT
from tests.factories import EXAMPLE_ANSWER, make_card, make_record, make_target


def _answer(card_patch: dict[str, Any] | None = None, **top: Any) -> str:
    """EXAMPLE_ANSWER with some keys replaced (for building invalid answers)."""
    data = json.loads(EXAMPLE_ANSWER)
    data["card"].update(card_patch or {})
    data.update(top)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def test_extra_key_rejected() -> None:
    with pytest.raises(TargetParseError, match="Extra inputs"):
        parse_target(_answer({"volume_m3": 80}))
    with pytest.raises(TargetParseError, match="Extra inputs"):
        parse_target(_answer(comment="ok"))
    with pytest.raises(ValidationError):
        make_card(volume_m3=80)


def test_missing_key_rejected_even_if_null() -> None:
    data = json.loads(EXAMPLE_ANSWER)
    del data["card"]["temperature_c"]
    with pytest.raises(TargetParseError, match="temperature_c: Field required"):
        parse_target(json.dumps(data, separators=(",", ":")))


def test_place_needs_city_and_region() -> None:
    assert Place(city="Казань", region="Республика Татарстан").region == "Республика Татарстан"
    with pytest.raises(ValidationError):
        Place(city="Казань")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Place(city="Houston", state="TX")  # type: ignore[call-arg]


def test_blank_strings_rejected() -> None:
    for blank in ("", "   "):
        with pytest.raises(ValidationError):
            Place(city=blank, region="Республика Татарстан")
        with pytest.raises(ValidationError):
            Place(city="Казань", region=blank)
    with pytest.raises(ValidationError):
        make_card(shipper_name=" ")


def test_serialize_parse_roundtrip() -> None:
    target = make_target(
        make_card(weight_total=Quantity(value=12.6, unit="t"), equipment_type=None),
    )
    assert parse_target(serialize_target(target)) == target


def test_serialize_is_canonical_and_cyrillic_not_escaped() -> None:
    assert serialize_target(make_target()) == EXAMPLE_ANSWER
    text = serialize_target(make_target(make_card(shipper_name="ООО «Ромашка»")))
    assert '"shipper_name":"ООО «Ромашка»"' in text
    assert '"origin":{"city":"Пермь","region":"Пермский край"}' in text
    assert "\\u" not in text


def test_quantity_int_stays_int() -> None:
    def weight_text(value: float) -> str:
        target = make_target(make_card(weight_total=Quantity(value=value, unit="t")))
        return json.dumps(json.loads(serialize_target(target))["card"]["weight_total"])

    assert Quantity(value=12592, unit="kg").value.__class__ is int
    assert weight_text(27761) == '{"value": 27761, "unit": "t"}'
    assert weight_text(12.6) == '{"value": 12.6, "unit": "t"}'
    assert weight_text(13.0) == '{"value": 13, "unit": "t"}'


def test_parse_rejects_code_fence() -> None:
    fenced = f"```json\n{EXAMPLE_ANSWER}\n```"
    with pytest.raises(TargetParseError, match="code fence"):
        parse_target(fenced)
    assert parse_target_lenient(fenced) == parse_target(EXAMPLE_ANSWER)
    assert parse_target_lenient(f"```{EXAMPLE_ANSWER}```") == parse_target(EXAMPLE_ANSWER)


def test_parse_rejects_surrounding_whitespace_and_prose() -> None:
    with pytest.raises(TargetParseError, match="whitespace"):
        parse_target(EXAMPLE_ANSWER + "\n")
    assert parse_target_lenient(f"  {EXAMPLE_ANSWER}\n") == parse_target(EXAMPLE_ANSWER)
    with pytest.raises(TargetParseError, match="invalid JSON"):
        parse_target_lenient(f"Вот карточка: {EXAMPLE_ANSWER}")


def test_parse_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    with pytest.raises(TargetParseError, match="duplicate keys"):
        parse_target(EXAMPLE_ANSWER[:-1] + ',"conflicts":[]}')
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(TargetParseError, match="not allowed"):
            parse_target(EXAMPLE_ANSWER.replace('"temperature_c":null', f'"temperature_c":{token}'))
    with pytest.raises(ValidationError):
        make_card(temperature_c=float("nan"))


@pytest.mark.parametrize(
    ("patch", "field"),
    [
        ({"pieces": "22"}, "pieces"),
        ({"pieces": True}, "pieces"),
        ({"pieces": 22.0}, "pieces"),
        ({"pieces": 0}, "pieces"),
        ({"weight_total": {"value": "12.6", "unit": "t"}}, "weight_total"),
        ({"weight_total": {"value": 0, "unit": "t"}}, "weight_total"),
        ({"weight_total": {"value": -1.5, "unit": "t"}}, "weight_total"),
        ({"weight_total": {"value": 12.6, "unit": "tons"}}, "weight_total"),
        ({"pickup_date": "2022-1-1"}, "pickup_date"),
        ({"pickup_date": "2022-01-01T00:00:00"}, "pickup_date"),
        ({"pickup_date": 1640995200}, "pickup_date"),
        ({"equipment_type": "flatbed"}, "equipment_type"),
        ({"cargo_category": "Retail"}, "cargo_category"),
    ],
)
def test_values_are_not_coerced(patch: dict[str, Any], field: str) -> None:
    with pytest.raises(TargetParseError, match=f"card.{field}"):
        parse_target(_answer(patch))


def test_conflict_needs_two_distinct_values() -> None:
    assert Conflict(field="pieces", values=[18, 20]).values == [18, 20]
    with pytest.raises(ValidationError):
        Conflict(field="pieces", values=[18])
    with pytest.raises(ValidationError, match="distinct"):
        Conflict(field="pieces", values=[18, 18])
    # 13 and 13.0 serialize identically, so they are the same value.
    with pytest.raises(ValidationError, match="distinct"):
        Conflict(field="pieces", values=[13, 13.0])
    same_weight = [{"value": 13, "unit": "t"}, {"value": 13.0, "unit": "t"}]
    with pytest.raises(ValidationError, match="distinct"):
        Conflict(field="weight_total", values=same_weight)
    with pytest.raises(ValidationError, match="distinct"):
        Conflict(field="origin", values=[[1, 2], [1.0, 2.0]])


def test_target_with_conflicts_roundtrips() -> None:
    weights = [{"value": 12.6, "unit": "t"}, {"value": 14.0, "unit": "t"}]
    target = make_target(
        make_card(weight_total=None, pieces=None),
        conflicts=[
            Conflict(field="weight_total", values=weights),
            Conflict(field="pieces", values=[18, 20.0]),
        ],
    )
    text = serialize_target(target)
    assert '"values":[{"value":12.6,"unit":"t"},{"value":14,"unit":"t"}]' in text
    assert '"values":[18,20]' in text
    assert serialize_target(parse_target(text)) == text
    with pytest.raises(ValidationError):
        Conflict(field="volume", values=[1, 2])  # type: ignore[arg-type]


def test_fieldname_literal_matches_card_fields() -> None:
    assert get_args(FieldName) == tuple(ShipmentCard.model_fields)


def test_target_schema_versions_are_supported() -> None:
    assert CARD_SCHEMA_VERSION == "card_v1"
    assert set(get_args(TargetSchemaVersion)) <= supported_versions("target")
    assert "sft_record_v1" in supported_versions("sft_dataset")


def test_record_jsonl_roundtrip() -> None:
    record = make_record()
    line = record.model_dump_json()
    assert "\n" not in line
    assert SFTRecord.model_validate_json(line) == record
    assert json.loads(line)["variant"]["request_date"] == "2021-12-28"


def test_record_roles_are_exactly_system_user_assistant() -> None:
    record = make_record()
    system, user, assistant = record.messages
    for messages in ([system, user], [user, system, assistant], [system, user, assistant, user]):
        with pytest.raises(ValidationError, match="roles must be exactly"):
            make_record(messages=messages)


def test_record_fields_are_validated() -> None:
    assert make_record(task="any_future_task").task == "any_future_task"  # open set (D-040)
    assert make_record(split=None).split is None
    for bad in (
        {"task": ""},
        {"schema_version": "card_v2"},
        {"split": "holdout"},
        {"language": "de"},
        {"id": " "},
    ):
        with pytest.raises(ValidationError):
            make_record(**bad)


def test_python_mode_needs_date_objects() -> None:
    """Strict models: JSON goes through model_validate_json, not json.loads + model_validate."""
    variant = make_record().variant
    data = json.loads(variant.model_dump_json())
    with pytest.raises(ValidationError, match="request_date"):
        VariantInfo.model_validate(data)
    assert VariantInfo.model_validate_json(json.dumps(data)) == variant
    assert variant.request_date == date(2021, 12, 28)


def test_record_messages_are_generation_messages() -> None:
    assert make_record().messages[0] == Message(role="system", content="system prompt")


def test_data_spec_examples_are_canonical_and_consistent() -> None:
    doc = (REPO_ROOT / "docs" / "DATA_SPEC.md").read_text(encoding="utf-8")
    answers = [block for block in re.findall(r"```json\n(.*?)\n```", doc, re.DOTALL)
               if block.startswith('{"card"')]  # fmt: skip
    assert len(answers) == 3
    assert EXAMPLE_ANSWER in answers
    for answer in answers:
        target = parse_target(answer)
        assert serialize_target(target) == answer
        assert check_target_consistency(target) == []


@pytest.mark.parametrize(
    ("text", "stage"),
    [
        (EXAMPLE_ANSWER + " ", "format"),
        (f"```{EXAMPLE_ANSWER}```", "format"),
        ("{", "format"),
        ('{"a":1,"a":2}', "format"),
        ("[]", "schema"),
        ('{"card":null,"missing_fields":[],"conflicts":[]}', "schema"),
    ],
)
def test_parse_error_tells_format_from_schema(text: str, stage: str) -> None:
    with pytest.raises(TargetParseError) as caught:
        parse_target(text)
    assert caught.value.stage == stage
