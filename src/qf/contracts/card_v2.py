"""Output contract `card_v2`: equipment types and special conditions (D-080…D-086).

Next to `card_v1`, which stays unchanged (plan C.7). Differences from card_v1:

- `equipment_type` has eight types, `cargo_category` adds `machinery` (D-082);
- `temperature_c` is gone: the temperature regime is one of the special conditions;
- `special_conditions` lists the conditions stated in the request, at most one per kind; an
  empty list means none (D-081, D-084). "Special conditions: yes/no" is computed by code.

Only numbers and values from closed lists, no free text. Values may be empty (`methods: []`,
both temperature bounds null): a condition named without values is a missing field (D-084).
Order is not part of the shape: the canonical order of conditions and list items is a
consistency rule of `qf.domain` (as the order of `missing_fields` in card_v1).
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Any, Final, Literal

from pydantic import Field, field_validator, model_validator

from qf.contracts._model import ContractModel, NonEmptyStr
from qf.contracts.card_v1 import Place, Quantity, _int_if_integral

__all__ = [
    "CONDITION_KINDS",
    "SCHEMA_VERSION",
    "CargoCategory",
    "ConditionKind",
    "Conflict",
    "ConflictField",
    "EquipmentType",
    "ExtractionTarget",
    "FieldName",
    "Length",
    "LengthUnit",
    "MissingField",
    "OversizeCondition",
    "PackagingCondition",
    "PackagingType",
    "SecuringCondition",
    "SecuringMethod",
    "SensorParameter",
    "SensorsCondition",
    "ShipmentCard",
    "SpecialCondition",
    "TemperatureCondition",
]

SCHEMA_VERSION: Final = "card_v2"

EquipmentType = Literal[
    "tent", "van", "reefer", "isotherm", "container", "lowbed", "mega", "flatbed"
]  # fmt: skip
CargoCategory = Literal[
    "general", "retail", "consumer_goods", "food_beverage", "automotive", "electronics",
    "machinery",
]  # fmt: skip
ConditionKind = Literal["temperature", "securing", "packaging", "oversize", "sensors"]
CONDITION_KINDS: Final[tuple[ConditionKind, ...]] = (
    "temperature", "securing", "packaging", "oversize", "sensors",
)  # fmt: skip
"""Canonical order of the conditions in a card and of their names in `missing_fields`."""
SecuringMethod = Literal["straps", "chains", "wheel_chocks", "anti_slip_mats", "load_bars"]
PackagingType = Literal["crate", "stretch_film", "moisture_protection", "shock_protection"]
SensorParameter = Literal["temperature", "humidity", "pressure", "tilt", "shock", "door_opening"]
LengthUnit = Literal["m", "cm"]

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
    "special_conditions",
]
"""Names of the `ShipmentCard` fields in declaration order (a test keeps the two in sync)."""
ConflictField = Literal[
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
]
"""Fields that can hold a conflict: all but `special_conditions` (D-084)."""
MissingField = Literal[
    "origin",
    "destination",
    "pickup_date",
    "equipment_type",
    "pieces",
    "weight_total",
    "special_conditions.temperature",
    "special_conditions.securing",
    "special_conditions.packaging",
    "special_conditions.oversize",
    "special_conditions.sensors",
]
"""Names that `missing_fields` may hold, in their canonical order (D-084)."""

PositiveNumber = Annotated[int, Field(gt=0)] | Annotated[float, Field(gt=0)]


def _unique(values: list[str]) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError("values must not repeat")
    return values


class Length(ContractModel):
    """A length exactly as written in the text; conversion to metres is done by code."""

    value: PositiveNumber
    unit: LengthUnit


class TemperatureCondition(ContractModel):
    """Keep the cargo within a range, °C: «+2…+6» → 2 and 6, «не выше +5» → null and 5."""

    kind: Literal["temperature"]
    min_c: float | None
    max_c: float | None

    @model_validator(mode="after")
    def _ordered(self) -> TemperatureCondition:
        if self.min_c is not None and self.max_c is not None and self.min_c > self.max_c:
            raise ValueError("min_c must not exceed max_c")
        return self


class SecuringCondition(ContractModel):
    kind: Literal["securing"]
    methods: list[SecuringMethod]

    @field_validator("methods")
    @classmethod
    def _methods_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)


class PackagingCondition(ContractModel):
    kind: Literal["packaging"]
    types: list[PackagingType]

    @field_validator("types")
    @classmethod
    def _types_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)


class OversizeCondition(ContractModel):
    """Out-of-gauge cargo: the dimensions stated in the text."""

    kind: Literal["oversize"]
    length: Length | None
    width: Length | None
    height: Length | None


class SensorsCondition(ContractModel):
    """Monitor and record parameters with sensors («с термописцем» → temperature). Keeping a
    temperature range is a `TemperatureCondition`; a request may state both."""

    kind: Literal["sensors"]
    parameters: list[SensorParameter]

    @field_validator("parameters")
    @classmethod
    def _parameters_unique(cls, values: list[str]) -> list[str]:
        return _unique(values)


SpecialCondition = Annotated[
    TemperatureCondition | SecuringCondition | PackagingCondition | OversizeCondition
    | SensorsCondition,
    Field(discriminator="kind"),
]  # fmt: skip


class ShipmentCard(ContractModel):
    """All keys are always present; an unknown value is null, no conditions is `[]`.

    `cargo_category` is a proxy, as in card_v1. `equipment_type` and the conditions of the
    synthetic data are assigned by a versioned table (D-083), not taken from the source.
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
    special_conditions: list[SpecialCondition]

    @field_validator("special_conditions")
    @classmethod
    def _one_per_kind(cls, conditions: list[SpecialCondition]) -> list[SpecialCondition]:
        kinds = [condition.kind for condition in conditions]
        if len(set(kinds)) != len(kinds):
            raise ValueError("at most one condition of each kind")
        return conditions


class Conflict(ContractModel):
    """A card field stated in the text with different values; the card holds null for it."""

    field: ConflictField
    values: list[Any] = Field(min_length=2)

    @field_validator("values")
    @classmethod
    def _values_distinct(cls, values: list[Any]) -> list[Any]:
        keys = [json.dumps(_int_if_integral(v), sort_keys=True, default=str) for v in values]
        if len(set(keys)) != len(keys):
            raise ValueError("conflict values must be distinct")
        return values


class ExtractionTarget(ContractModel):
    """The whole assistant answer: one JSON object, no markdown around it."""

    card: ShipmentCard
    missing_fields: list[MissingField]
    conflicts: list[Conflict]
