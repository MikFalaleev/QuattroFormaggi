"""Facts about one shipment for `card_v2` (sub-step V2, D-083): `load_facts_v2`.

Derived from `load_facts_v1` (which stays unchanged) by a versioned table
(`configs/data/equipment_conditions_v1.yaml`) that assigns the equipment type, the special
conditions and, for special machinery, the cargo category and the number of pieces. The facts
are complete: every condition has its values; the request generator (V3) may leave values out
of the text, never the facts. Read a line with `LoadFactsV2.model_validate_json(line)`.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Final

from pydantic import Field, field_validator, model_validator

from qf.contracts._model import ContractModel, NonEmptyStr
from qf.contracts.card_v1 import Place
from qf.contracts.card_v2 import CargoCategory, EquipmentType, SpecialCondition
from qf.contracts.facts import LoadFacts

__all__ = ["LOAD_FACTS_V2_SCHEMA_VERSION", "AnyLoadFacts", "LoadFactsV2"]

LOAD_FACTS_V2_SCHEMA_VERSION: Final = "load_facts_v2"


class LoadFactsV2(ContractModel):
    """`LoadFacts` in card_v2 terms plus the special conditions and the table profile used.

    Completeness (a reefer has a temperature, a lowbed trailer all three dimensions, no
    condition without values) is checked by the stage with the card_v2 rules of `qf.domain`,
    so the rule lives in one place.
    """

    load_id: NonEmptyStr
    route_id: NonEmptyStr
    customer_id: NonEmptyStr
    shipper_name: NonEmptyStr
    cargo_category: CargoCategory
    equipment_type: EquipmentType
    pieces: Annotated[int, Field(ge=1)]
    weight_lbs: Annotated[int, Field(gt=0)]
    origin: Place
    destination: Place
    pickup_date: date
    delivery_date: date
    load_month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    special_conditions: list[SpecialCondition]
    profile: NonEmptyStr  # "<source cargo category>/<profile name>" of the table

    @field_validator("special_conditions")
    @classmethod
    def _one_per_kind(cls, conditions: list[SpecialCondition]) -> list[SpecialCondition]:
        kinds = [condition.kind for condition in conditions]
        if len(set(kinds)) != len(kinds):
            raise ValueError("at most one condition of each kind")
        return conditions

    @model_validator(mode="after")
    def _delivery_not_before_pickup(self) -> LoadFactsV2:
        if self.delivery_date < self.pickup_date:
            raise ValueError(f"delivery {self.delivery_date} is before pickup {self.pickup_date}")
        return self


AnyLoadFacts = LoadFacts | LoadFactsV2
"""Facts of either version: the input of the request generator (card_v1 / card_v2, D-094)."""
