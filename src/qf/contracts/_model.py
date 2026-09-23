"""Base model and shared field types of the versioned data contracts (card_v1, sft_record_v1)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ContractModel", "NonEmptyStr"]

NonEmptyStr = Annotated[str, Field(min_length=1, pattern=r"\S")]
"""A string with at least one non-whitespace character."""


class ContractModel(BaseModel):
    """Immutable, closed and strict (D-041).

    Strict: no type coercion, so `true` or `"22"` is not a number and a date must be an ISO
    `YYYY-MM-DD` string. Parse JSON with `Model.model_validate_json(text)`; `model_validate`
    on a `json.loads` result fails on dates. NaN and infinities are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
