"""Types of the request generator: a draft, rendered evidence and the rendered request (step 6).

A `RequestDraft` is the plan of one record before rendering; hard cases change the draft and a
template family renders it. Every gold value comes from `RenderedField` evidence in the text.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field

from qf.contracts._model import ContractModel
from qf.contracts.card_v1 import ExtractionTarget, FieldName, WeightUnit
from qf.contracts.card_v2 import ConditionKind, LengthUnit
from qf.contracts.card_v2 import ExtractionTarget as ExtractionTargetV2
from qf.contracts.card_v2 import FieldName as FieldNameV2
from qf.contracts.records import (
    HardCaseName,
    Language,
    OodReason,
    TargetSchemaVersion,
    VariantInfo,
)

__all__ = ["AnyFieldName", "Dimension", "RenderedField", "RenderedRequest", "RequestDraft"]

AnyFieldName = FieldName | FieldNameV2
"""A card field of card_v1 or card_v2 (the generator renders both, D-094)."""
Dimension = Literal["length", "width", "height"]


class RenderedField(ContractModel):
    """One mention of a card field in the text: the exact substring and the value it denotes."""

    text: str = Field(min_length=1)
    field: AnyFieldName
    gold_value: Any


class RequestDraft(ContractModel):
    """What to render for one load.

    `conflicts` maps a field to the second, different source value written in the text
    (weight_total: pounds; pieces: count). `distractors` holds numbers that are not card fields
    (rate in rubles, dock, request number) and must never reach the card.

    The fields after `ood_reason` are card_v2 only (sub-step V3, D-094); their defaults keep
    card_v1 drafts as they were. `conditions_dropped` are left out of the text,
    `conditions_without_values` are named without their values, `oversize_dims` lists the
    dimensions written (None: all the facts have) and `conditions_scattered` spreads the
    conditions over the text instead of one group.
    """

    language: Language
    request_date: date
    weight_unit: WeightUnit
    weight_mode: Literal["total", "per_piece"]
    date_style: Literal["iso", "text", "relative"]
    city_lang: Language
    dropped_fields: list[FieldName]
    conflicts: dict[FieldName, int]
    distractors: dict[str, int]
    hard_cases: list[HardCaseName]
    ood_reason: OodReason | None
    schema_version: TargetSchemaVersion = "card_v1"
    length_unit: LengthUnit = "m"
    conditions_dropped: list[ConditionKind] = Field(default_factory=list)
    conditions_without_values: list[ConditionKind] = Field(default_factory=list)
    oversize_dims: list[Dimension] | None = None
    conditions_scattered: bool = False


class RenderedRequest(ContractModel):
    """Request text with the evidence for every field and the gold answer built from it."""

    text: str = Field(min_length=1)
    evidence: dict[AnyFieldName, list[str]]
    target: ExtractionTarget | ExtractionTargetV2
    variant: VariantInfo
