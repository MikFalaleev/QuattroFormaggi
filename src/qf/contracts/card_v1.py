"""Output contract `card_v1`: the answer the model returns for a freight request (plan B.3).

A new card version is a new module next to this one (`card_v2.py`); this file is not changed
once records using it exist (plan C.7). Values are checked only for shape here; units are
converted, missing fields computed and values validated by `qf.domain`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Any, Final, Literal

from pydantic import Field, field_validator

from qf.contracts._model import ContractModel, NonEmptyStr

__all__ = [
    "SCHEMA_VERSION",
    "CargoCategory",
    "Conflict",
    "EquipmentType",
    "ExtractionTarget",
    "FieldName",
    "Place",
    "Quantity",
    "ShipmentCard",
    "WeightUnit",
]

SCHEMA_VERSION: Final = "card_v1"

EquipmentType = Literal["dry_van", "reefer"]
CargoCategory = Literal[
    "general", "retail", "consumer_goods", "food_beverage", "automotive", "electronics"
]
WeightUnit = Literal["kg", "t", "lb"]
FieldName = Literal[
    "shipper_name",
    "cargo_category",
    "equipment_type",
    "pieces",
    "weight_total",
    "weight_per_piece",
    "origin",
    "destination",
    "pickup_date",
    "delivery_date",
    "temperature_c",
]
"""Names of the `ShipmentCard` fields in declaration order (a test keeps the two in sync)."""

PositiveNumber = Annotated[int, Field(gt=0)] | Annotated[float, Field(gt=0)]


class Quantity(ContractModel):
    """A mass exactly as written in the text; conversion to kg is done by code (D-004).

    `value` keeps its JSON type: 27761 stays an int and 12.6 a float.
    """

    value: PositiveNumber
    unit: WeightUnit


class Place(ContractModel):
    """Canonical Russian city and region names, e.g. Казань, Республика Татарстан (D-048).

    For a federal city the region equals the city (Москва, Москва).
    """

    city: NonEmptyStr
    region: NonEmptyStr


class ShipmentCard(ContractModel):
    """All keys are always present; an unknown value is null.

    `cargo_category` is a proxy: it comes from the customer's primary freight type, not from
    the cargo itself. `temperature_c` is always null in card_v1 data (the source has none).
    """

    shipper_name: NonEmptyStr | None
    cargo_category: CargoCategory | None
    equipment_type: EquipmentType | None
    pieces: Annotated[int, Field(ge=1)] | None
    weight_total: Quantity | None
    weight_per_piece: Quantity | None
    origin: Place | None
    destination: Place | None
    pickup_date: date | None
    delivery_date: date | None
    temperature_c: float | None


class Conflict(ContractModel):
    """A field stated in the text with different values; the card holds null for it."""

    field: FieldName
    values: list[Any] = Field(min_length=2)

    @field_validator("values")
    @classmethod
    def _values_distinct(cls, values: list[Any]) -> list[Any]:
        keys = [json.dumps(_int_if_integral(v), sort_keys=True, default=str) for v in values]
        if len(set(keys)) != len(keys):
            raise ValueError("conflict values must be distinct")
        return values


def _int_if_integral(value: Any) -> Any:
    """13.0 and 13 are the same value: the canonical serialization writes both as 13."""
    if isinstance(value, dict):
        return {key: _int_if_integral(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_int_if_integral(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class ExtractionTarget(ContractModel):
    """The whole assistant answer: one JSON object, no markdown around it."""

    card: ShipmentCard
    missing_fields: list[FieldName]
    conflicts: list[Conflict]
