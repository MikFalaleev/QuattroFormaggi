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
from qf.contracts.records import HardCaseName, Language, OodReason, VariantInfo

__all__ = ["RenderedField", "RenderedRequest", "RequestDraft"]


class RenderedField(ContractModel):
    """One mention of a card field in the text: the exact substring and the value it denotes."""

    text: str = Field(min_length=1)
    field: FieldName
    gold_value: Any


class RequestDraft(ContractModel):
    """What to render for one load.

    `conflicts` maps a field to the second, different source value written in the text
    (weight_total: pounds; pieces: count). `distractors` holds numbers that are not card fields
    (rate in rubles, dock, request number) and must never reach the card.
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


class RenderedRequest(ContractModel):
    """Request text with the evidence for every field and the gold answer built from it."""

    text: str = Field(min_length=1)
    evidence: dict[FieldName, list[str]]
    target: ExtractionTarget
    variant: VariantInfo
