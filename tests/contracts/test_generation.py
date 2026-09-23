from __future__ import annotations

import pytest
from pydantic import ValidationError

from qf.contracts import GenerationRequest, GenerationResult, Message


def test_request_requires_messages_and_positive_budget() -> None:
    message = Message(role="user", content="Нужно перевезти 12,6 т из Хьюстона")
    request = GenerationRequest(messages=[message], max_tokens=768, temperature=0.0)
    assert request.json_schema is None
    with pytest.raises(ValidationError):
        GenerationRequest(messages=[], max_tokens=768, temperature=0.0)
    with pytest.raises(ValidationError):
        GenerationRequest(messages=[message], max_tokens=0, temperature=0.0)


def test_message_rejects_unknown_role_and_fields() -> None:
    with pytest.raises(ValidationError):
        Message(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Message(role="user", content="x", name="bob")  # type: ignore[call-arg]


def test_result_ok_reflects_error() -> None:
    assert GenerationResult(text="{}", latency_s=0.5).ok
    assert not GenerationResult(text="", latency_s=2.0, error="timeout").ok
