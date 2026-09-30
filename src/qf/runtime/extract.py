"""`extract`: request text -> shipment card, checked by code (step 18).

The use case depends only on the `GenerationBackend` port: which backend answers (LM Studio,
llama-server, ...) is chosen by the config through `qf.cli.wiring`. The messages are built by
`qf.domain.build_messages`, the same code as in the training data. The model's answer is parsed
strictly; its `missing_fields` are replaced by the code's rule, and the values are checked
independently (`qf.domain.check_card_values`). Nothing here retries or repairs an answer.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from qf.common import ComponentConfig, StrictConfig
from qf.contracts import CARD_V2_SCHEMA_VERSION, GenerationBackend, GenerationRequest, Language
from qf.domain import (
    SHIPMENT_EXTRACTION,
    TargetParseError,
    build_messages,
    check_card_values,
    effective_pieces,
    get_target_schema,
    get_task,
    system_prompt_hash,
    total_weight_kg,
)

__all__ = [
    "ASSUMPTION_DEFAULT_PIECES",
    "ExtractConfig",
    "ExtractionResult",
    "ExtractSettings",
    "detect_language",
    "extract_card",
]

ASSUMPTION_DEFAULT_PIECES: Final = "pieces=1 по умолчанию, уточнить"
_CYRILLIC: Final = re.compile(r"[а-яёА-ЯЁ]")

Status = Literal["ok", "invalid_output", "backend_error"]


class ExtractSettings(StrictConfig):
    """How the model is asked: the gate mode is temperature 0 without the JSON schema."""

    max_tokens: int = Field(default=768, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    json_schema: bool = False  # send the answer's JSON schema (constrained decoding)


class ExtractConfig(StrictConfig):
    """`configs/runtime/extract.yaml`."""

    backend: ComponentConfig
    schema_version: str = CARD_V2_SCHEMA_VERSION
    task: str = SHIPMENT_EXTRACTION.name
    settings: ExtractSettings = Field(default_factory=ExtractSettings)


class Derived(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weight_total_kg: float | None
    pieces: int | None


class ExtractionResult(BaseModel):
    """The checked answer. `card` is None unless `status == "ok"`; `raw_text` is always kept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    card: dict[str, Any] | None
    missing_fields: list[str]
    conflicts: list[dict[str, Any]]
    derived: Derived
    assumptions: list[str]
    warnings: list[str]
    review_required: bool
    model_id: str
    prompt_hash: str
    schema_version: str
    checked_at: str
    raw_text: str
    error: str | None = None


def detect_language(text: str) -> Language:
    """`ru` if the text has Cyrillic letters, else `en`."""
    return "ru" if _CYRILLIC.search(text) else "en"


def extract_card(
    request_text: str,
    request_date: date,
    language: Language,
    backend: GenerationBackend,
    *,
    settings: ExtractSettings | None = None,
    schema_version: str = CARD_V2_SCHEMA_VERSION,
    task: str = SHIPMENT_EXTRACTION.name,
    now: datetime | None = None,
) -> ExtractionResult:
    settings = settings or ExtractSettings()
    schema = get_target_schema(schema_version)
    prompt_version = get_task(task).prompt_for(schema_version)
    request = GenerationRequest(
        messages=build_messages(request_text, request_date, language, prompt_version),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        json_schema=schema.json_schema() if settings.json_schema else None,
    )
    generated = backend.generate(request)
    common: dict[str, Any] = {
        "model_id": backend.model_id(),
        "prompt_hash": system_prompt_hash(prompt_version),
        "schema_version": schema_version,
        "checked_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "raw_text": generated.text,
    }
    failed = {"card": None, "missing_fields": [], "conflicts": [], "assumptions": [],
              "derived": Derived(weight_total_kg=None, pieces=None),
              "review_required": True, **common}  # fmt: skip
    if not generated.ok:
        return ExtractionResult(status="backend_error", warnings=["backend_error"],
                                error=generated.error, **failed)  # fmt: skip
    try:
        target = schema.parse(generated.text)
    except TargetParseError as exc:
        return ExtractionResult(status="invalid_output", warnings=[f"invalid_output:{exc.stage}"],
                                error=str(exc), **failed)  # fmt: skip

    card = target.card
    warnings = []
    expected = list(schema.missing_fields(card))
    if list(target.missing_fields) != expected:
        warnings.append("missing_mismatch")
    # the rule of the code replaces the model's list before the other consistency checks
    checked = target.model_copy(update={"missing_fields": expected})
    warnings += [f"inconsistent:{problem}" for problem in schema.check_consistency(checked)]
    warnings += check_card_values(card, request_date)
    assumptions = []
    if card.pieces is None:
        assumptions.append(ASSUMPTION_DEFAULT_PIECES)
    return ExtractionResult(
        status="ok",
        card=card.model_dump(mode="json"),
        missing_fields=[str(name) for name in expected],
        conflicts=[c.model_dump(mode="json") for c in target.conflicts],
        derived=Derived(weight_total_kg=total_weight_kg(card), pieces=effective_pieces(card)),
        assumptions=assumptions,
        warnings=warnings,
        review_required=bool(warnings),
        **common,
    )
