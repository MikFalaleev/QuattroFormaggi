"""Facts about one shipment: the only input of the request generator (plan step 5).

The contract is neutral: any source (this dataset, real requests, another TMS) produces it.
Stored as JSONL, one record per line; read a line with `LoadFacts.model_validate_json(line)`.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Final

from pydantic import Field, model_validator

from qf.contracts._model import ContractModel, NonEmptyStr
from qf.contracts.card_v1 import CargoCategory, EquipmentType, Place

__all__ = ["LOAD_FACTS_SCHEMA_VERSION", "LoadFacts"]

LOAD_FACTS_SCHEMA_VERSION: Final = "load_facts_v1"


class LoadFacts(ContractModel):
    """What is known about one shipment, already in card terms (Russian places, card enums).

    Deliberately without revenue, surcharges, rates or distances: they are not in card_v1 and
    must never reach a gold answer.
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

    @model_validator(mode="after")
    def _delivery_not_before_pickup(self) -> LoadFacts:
        if self.delivery_date < self.pickup_date:
            raise ValueError(f"delivery {self.delivery_date} is before pickup {self.pickup_date}")
        return self
