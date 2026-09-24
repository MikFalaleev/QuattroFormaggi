"""What the lower-bound baselines share (D-109): the request date of the user message and the
canonical answer of a card, built by the schema registry like any gold answer."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Final

from pydantic import ValidationError

from qf.common import QFError
from qf.contracts import GenerationRequest, GenerationResult
from qf.domain import get_target_schema

__all__ = ["REQUEST_DATE", "answer_text", "empty_card", "refuse", "request_date", "user_text"]

REQUEST_DATE: Final = re.compile(r"^(?:Дата запроса|Request date):\s*(\d{4}-\d{2}-\d{2})", re.M)


def refuse(req: GenerationRequest, name: str) -> GenerationResult | None:
    """The error result for a request that does not end with the user message (a baseline
    never sees the gold answer), None otherwise."""
    last = req.messages[-1].role
    if last != "user":
        return GenerationResult(text="", latency_s=0.0,
                                error=f"{name}: last message is {last}, not user")  # fmt: skip
    return None


def user_text(req: GenerationRequest) -> str:
    return req.messages[-1].content


def request_date(text: str) -> date | None:
    match = REQUEST_DATE.search(text)
    try:
        return date.fromisoformat(match.group(1)) if match else None
    except ValueError:
        return None  # an impossible date: relative dates stay unknown


def empty_card(schema_version: str) -> dict[str, Any]:
    """Every field null, no special conditions: the card of a model that knows nothing."""
    schema = get_target_schema(schema_version)
    card: dict[str, Any] = dict.fromkeys(schema.card_fields)
    if "special_conditions" in card:
        card["special_conditions"] = []
    return card


def answer_text(schema_version: str, card: dict[str, Any]) -> str:
    """The canonical answer of `card`: missing fields by the rule, no conflicts. A card that the
    schema refuses loses its special conditions, then everything (never an invalid answer)."""
    schema = get_target_schema(schema_version)
    fallbacks = [card, {**card, "special_conditions": []} if "special_conditions" in card
                 else card, empty_card(schema_version)]  # fmt: skip
    for candidate in fallbacks:
        try:
            model = schema.card_model.model_validate_json(json.dumps(candidate, default=str))
        except ValidationError:
            continue
        model = schema.canonicalize(model)
        target = schema.target_model.model_validate_json(json.dumps(
            {"card": model.model_dump(mode="json"),
             "missing_fields": list(schema.missing_fields(model)), "conflicts": []}))  # fmt: skip
        text: str = schema.serialize(target)
        return text
    raise QFError(f"{schema_version}: even the empty card is refused")
