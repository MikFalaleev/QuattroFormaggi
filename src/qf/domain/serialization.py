"""The only way to turn an answer into assistant text and back (plan step 4).

`serialize_target` defines the canonical form: compact JSON, keys in field declaration order,
Cyrillic not escaped, integral numbers without `.0`. `parse_target` is strict: anything other
than one bare JSON object of the card_v1 schema raises `TargetParseError`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import ValidationError

from qf.common import QFError, format_validation_error
from qf.contracts import ExtractionTarget
from qf.domain.units import normalize_number

__all__ = ["TargetParseError", "parse_target", "parse_target_lenient", "serialize_target"]

_CODE_FENCE = re.compile(r"\A```[A-Za-z]*\s*(?P<body>.*?)\s*```\Z", re.DOTALL)


class TargetParseError(QFError):
    """The text is not a valid card_v1 answer.

    `stage` is "format" if the text is not one bare, well-formed JSON object and "schema" if
    it is, but does not match card_v1 (step 9 counts the two separately).
    """

    def __init__(self, message: str, *, stage: Literal["format", "schema"]) -> None:
        super().__init__(message)
        self.stage = stage


def serialize_target(target: ExtractionTarget) -> str:
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
    """Strict parse: no code fence, no surrounding whitespace, no duplicate keys, no NaN."""
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
        return ExtractionTarget.model_validate_json(text)
    except ValidationError as exc:
        message = f"not a card_v1 answer: {format_validation_error(exc)}"
        raise TargetParseError(message, stage="schema") from exc


def parse_target_lenient(text: str) -> ExtractionTarget:
    """Strips surrounding whitespace and one code fence, then parses strictly.

    For field-level metrics only (step 9); `json_valid_rate` uses `parse_target`.
    """
    stripped = text.strip()
    fenced = _CODE_FENCE.match(stripped)
    return parse_target(fenced.group("body") if fenced else stripped)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")
