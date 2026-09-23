"""Builders of valid card_v1 objects and SFT records for tests (every card field is required)."""

from __future__ import annotations

from datetime import date
from typing import Any

from qf.contracts import (
    ExtractionTarget,
    Message,
    Place,
    Quantity,
    SFTRecord,
    ShipmentCard,
    VariantInfo,
)
from qf.domain import compute_missing_fields, serialize_target

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
