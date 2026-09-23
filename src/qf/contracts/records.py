"""SFT record format `sft_record_v1`: one JSON line per training or evaluation example (step 4).

Read a record with `SFTRecord.model_validate_json(line)`: the models are strict (see
`ContractModel`). `Message` is shared with the generation contract (`qf.contracts.generation`).
"""

from __future__ import annotations

from datetime import date
from typing import Final, Literal

from pydantic import Field, field_validator

from qf.contracts._model import ContractModel, NonEmptyStr
from qf.contracts.card_v1 import FieldName, WeightUnit
from qf.contracts.generation import Message

__all__ = [
    "RECORD_ROLES",
    "SFT_RECORD_SCHEMA_VERSION",
    "HardCaseName",
    "Language",
    "OodReason",
    "SFTRecord",
    "SplitName",
    "TargetSchemaVersion",
    "TaskName",
    "VariantInfo",
]

SFT_RECORD_SCHEMA_VERSION: Final = "sft_record_v1"
RECORD_ROLES: Final = ("system", "user", "assistant")

TaskName = str
"""Open set: a name is valid if it is registered in `qf.domain.TASKS` (D-040)."""
HardCaseName = str
"""Open set: a name is valid if it is registered in `qf.data.HARD_CASES` (step 6)."""
TargetSchemaVersion = Literal["card_v1", "card_v2"]
"""Answer schemas a record may use. A new schema widens this Literal; old records stay valid."""
Language = Literal["ru", "en"]
SplitName = Literal["train", "val", "test", "test_ood", "bench"]
OodReason = Literal["route", "family", "both"]
"""Why a test_ood record is out of distribution: held-out route, held-out template family, both."""


class VariantInfo(ContractModel):
    """How the request text was rendered; used for slicing metrics and checking the generator."""

    weight_unit: WeightUnit | None
    weight_mode: Literal["total", "per_piece", "none"]
    dropped_fields: list[FieldName]
    hard_cases: list[HardCaseName]
    request_date: date
    date_style: Literal["iso", "text", "relative"]
    city_lang: Language
    ood_reason: OodReason | None


class SFTRecord(ContractModel):
    """One example: messages are exactly [system, user, assistant]; the assistant content is
    the canonical serialization of the target (checked by `qf.domain.check_record`)."""

    id: NonEmptyStr
    task: TaskName = Field(min_length=1)
    group_id: NonEmptyStr
    source_id: NonEmptyStr
    language: Language
    reviewed: bool
    synthetic: bool
    template_family: NonEmptyStr
    variant: VariantInfo
    schema_version: TargetSchemaVersion
    split: SplitName | None
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def _roles_in_order(cls, messages: list[Message]) -> list[Message]:
        roles = tuple(message.role for message in messages)
        if roles != RECORD_ROLES:
            raise ValueError(f"roles must be exactly {list(RECORD_ROLES)}, got {list(roles)}")
        return messages
