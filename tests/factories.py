"""Builders of valid card_v1 / card_v2 objects and SFT records for tests (every card field is
required)."""

from __future__ import annotations

from datetime import date
from typing import Any

from qf.contracts import (
    ExtractionTarget,
    ExtractionTargetV2,
    Message,
    Place,
    Quantity,
    SensorsCondition,
    SFTRecord,
    ShipmentCard,
    ShipmentCardV2,
    TemperatureCondition,
    VariantInfo,
)
from qf.domain import (
    compute_missing_fields,
    compute_missing_fields_v2,
    load_system_prompt,
    serialize_target,
)

# The example answer of IMPLEMENTATION_PLAN.md B.3, in its canonical serialization.
EXAMPLE_ANSWER = (
    '{"card":{"shipper_name":"National Retail","cargo_category":"retail",'
    '"equipment_type":"dry_van","pieces":22,"weight_total":{"value":12592,"unit":"kg"},'
    '"weight_per_piece":null,"origin":{"city":"Пермь","region":"Пермский край"},'
    '"destination":{"city":"Самара","region":"Самарская область"},"pickup_date":"2022-01-01",'
    '"delivery_date":"2022-01-02","temperature_c":null},"missing_fields":[],"conflicts":[]}'
)


def make_card(**overrides: Any) -> ShipmentCard:
    fields: dict[str, Any] = {
        "shipper_name": "National Retail",
        "cargo_category": "retail",
        "equipment_type": "dry_van",
        "pieces": 22,
        "weight_total": Quantity(value=12592, unit="kg"),
        "weight_per_piece": None,
        "origin": Place(city="Пермь", region="Пермский край"),
        "destination": Place(city="Самара", region="Самарская область"),
        "pickup_date": date(2022, 1, 1),
        "delivery_date": date(2022, 1, 2),
        "temperature_c": None,
    }
    fields.update(overrides)
    return ShipmentCard(**fields)


def make_target(card: ShipmentCard | None = None, **overrides: Any) -> ExtractionTarget:
    """A consistent target: missing_fields follow the rule unless overridden."""
    card = card if card is not None else make_card()
    fields: dict[str, Any] = {
        "card": card,
        "missing_fields": compute_missing_fields(card),
        "conflicts": [],
    }
    fields.update(overrides)
    return ExtractionTarget(**fields)


def make_record(answer: str | None = None, **overrides: Any) -> SFTRecord:
    fields: dict[str, Any] = {
        "id": "qf-train-LOAD00001-0",
        "task": "shipment_extraction",
        "group_id": "load:LOAD00001",
        "source_id": "yogape/logistics-operations@54e7d1d1a437:LOAD00001",
        "language": "ru",
        "reviewed": False,
        "synthetic": True,
        "template_family": "T1",
        "variant": VariantInfo(
            weight_unit="kg",
            weight_mode="total",
            dropped_fields=[],
            hard_cases=[],
            request_date=date(2021, 12, 28),
            date_style="iso",
            city_lang="ru",
            ood_reason=None,
        ),
        "schema_version": "card_v1",
        "split": "train",
        "messages": [
            Message(role="system", content="system prompt"),
            Message(role="user", content="Дата запроса: 2021-12-28\n\nНужна машина"),
            Message(
                role="assistant",
                content=answer if answer is not None else serialize_target(make_target()),
            ),
        ],
    }
    fields.update(overrides)
    return SFTRecord(**fields)


# The example answer of docs/PLAN_card_v2.md, section 4, in its canonical serialization.
EXAMPLE_ANSWER_V2 = (
    '{"card":{"shipper_name":"ООО Север","cargo_category":"food_beverage",'
    '"equipment_type":"reefer","pieces":18,"weight_total":{"value":12.6,"unit":"t"},'
    '"weight_per_piece":null,"origin":{"city":"Пермь","region":"Пермский край"},'
    '"destination":{"city":"Казань","region":"Республика Татарстан"},'
    '"pickup_date":"2024-03-12","delivery_date":"2024-03-14","special_conditions":['
    '{"kind":"temperature","min_c":2,"max_c":6},{"kind":"sensors","parameters":["temperature"]}'
    ']},"missing_fields":[],"conflicts":[]}'
)


def make_card_v2(**overrides: Any) -> ShipmentCardV2:
    """A reefer at +2…+6 °C with a temperature sensor, unless overridden."""
    fields: dict[str, Any] = {
        "shipper_name": "ООО Север",
        "cargo_category": "food_beverage",
        "equipment_type": "reefer",
        "pieces": 18,
        "weight_total": Quantity(value=12.6, unit="t"),
        "weight_per_piece": None,
        "origin": Place(city="Пермь", region="Пермский край"),
        "destination": Place(city="Казань", region="Республика Татарстан"),
        "pickup_date": date(2024, 3, 12),
        "delivery_date": date(2024, 3, 14),
        "special_conditions": [
            TemperatureCondition(kind="temperature", min_c=2, max_c=6),
            SensorsCondition(kind="sensors", parameters=["temperature"]),
        ],
    }
    fields.update(overrides)
    return ShipmentCardV2(**fields)


def make_target_v2(card: ShipmentCardV2 | None = None, **overrides: Any) -> ExtractionTargetV2:
    """A consistent card_v2 target: missing_fields follow the rule unless overridden."""
    card = card if card is not None else make_card_v2()
    fields: dict[str, Any] = {
        "card": card,
        "missing_fields": compute_missing_fields_v2(card),
        "conflicts": [],
    }
    fields.update(overrides)
    return ExtractionTargetV2(**fields)


def make_record_v2(answer: str | None = None, **overrides: Any) -> SFTRecord:
    """A card_v2 record with the system prompt of card_v2."""
    base = make_record()
    messages = [
        Message(role="system", content=load_system_prompt("system_extract_v2")),
        base.messages[1],
        Message(role="assistant",
                content=answer if answer is not None else serialize_target(make_target_v2())),
    ]  # fmt: skip
    fields: dict[str, Any] = {**base.model_dump(), "schema_version": "card_v2",
                              "variant": base.variant, "messages": messages}  # fmt: skip
    fields.update(overrides)
    return SFTRecord(**fields)
