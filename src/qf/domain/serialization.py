"""The only way to turn an answer into assistant text and back (plan step 4, D-086).

`serialize_target` defines the canonical form of every answer schema: compact JSON, keys in
field declaration order, Cyrillic not escaped, integral numbers without `.0`. `parse_target`
(card_v1) and `parse_target_v2` (card_v2) are strict: anything other than one bare JSON
object of the schema raises `TargetParseError`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from qf.common import QFError, format_validation_error
from qf.contracts import ExtractionTarget, ExtractionTargetV2
from qf.domain.units import normalize_number

__all__ = [
    "TargetParseError",
    "parse_target",
    "parse_target_lenient",
    "parse_target_lenient_v2",
    "parse_target_v2",
    "serialize_target",
]

TargetT = TypeVar("TargetT", bound=BaseModel)
_CODE_FENCE = re.compile(r"\A```[A-Za-z]*\s*(?P<body>.*?)\s*```\Z", re.DOTALL)


class TargetParseError(QFError):
    """The text is not a valid answer of the expected schema.

    `stage` is "format" if the text is not one bare, well-formed JSON object and "schema" if
    it is, but does not match the schema (step 9 counts the two separately).
    """

    def __init__(self, message: str, *, stage: Literal["format", "schema"]) -> None:
        super().__init__(message)
        self.stage = stage


def serialize_target(target: ExtractionTarget | ExtractionTargetV2) -> str:
    data = _normalize_numbers(target.model_dump(mode="json"))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _normalize_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _normalize_numbers(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize_numbers(value) for value in obj]
    if isinstance(obj, float):
        return normalize_number(obj)
    return obj


def parse_target(text: str) -> ExtractionTarget:
    """Strict card_v1 parse: no code fence, no surrounding whitespace, no duplicate keys, no
    NaN."""
    return _parse_strict(text, ExtractionTarget, "card_v1")


def parse_target_v2(text: str) -> ExtractionTargetV2:
    """Strict card_v2 parse, with the same format rules as `parse_target`."""
    return _parse_strict(text, ExtractionTargetV2, "card_v2")


def _parse_strict(text: str, model: type[TargetT], schema: str) -> TargetT:
    if text != text.strip():
        raise TargetParseError(
            "leading or trailing whitespace around the JSON object", stage="format"
        )
    if text.startswith("```"):
        raise TargetParseError("a code fence around the JSON object", stage="format")
    try:
        json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except ValueError as exc:  # includes json.JSONDecodeError
        raise TargetParseError(f"invalid JSON: {exc}", stage="format") from exc
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        message = f"not a {schema} answer: {format_validation_error(exc)}"
        raise TargetParseError(message, stage="schema") from exc


def parse_target_lenient(text: str) -> ExtractionTarget:
    """Strips surrounding whitespace and one code fence, then parses strictly.

    For field-level metrics only (step 9); `json_valid_rate` uses `parse_target`.
    """
    return parse_target(_unwrap(text))


def parse_target_lenient_v2(text: str) -> ExtractionTargetV2:
    """`parse_target_lenient` for card_v2."""
    return parse_target_v2(_unwrap(text))


def _unwrap(text: str) -> str:
    stripped = text.strip()
    fenced = _CODE_FENCE.match(stripped)
    return fenced.group("body") if fenced else stripped


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")
