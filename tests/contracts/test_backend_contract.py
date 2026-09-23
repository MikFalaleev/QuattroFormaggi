"""Contract test of every registered generation backend (plan C.8, step 9).

Each backend must be testable offline: `OFFLINE_CONFIGS` holds a configuration that needs no
network, GPU or model (for `openai_local`, step 10: a mock transport). A backend registered
without an entry here fails `test_every_backend_has_an_offline_config`.
"""

from __future__ import annotations

from typing import Any

import pytest

import qf.cli.wiring  # noqa: F401  (every registry and lazy entry)
from qf.backends import BACKENDS
from qf.cli.wiring import build
from qf.common import ComponentConfig, sha256_text
from qf.contracts import GenerationBackend, GenerationRequest, GenerationResult, Message

QUESTION = "Нужна машина из Перми в Самару"
ANSWER = '{"card": {}}'
OFFLINE_CONFIGS: dict[str, dict[str, Any]] = {
    "fake": {"responses": {sha256_text(QUESTION): ANSWER}},
}


def request(*messages: Message) -> GenerationRequest:
    return GenerationRequest(messages=list(messages), max_tokens=64, temperature=0.0)


def backend(name: str) -> GenerationBackend:
    instance = build(BACKENDS, ComponentConfig(name=name, **OFFLINE_CONFIGS[name]))
    assert isinstance(instance, GenerationBackend)
    return instance


def test_every_backend_has_an_offline_config() -> None:
    assert sorted(BACKENDS.names()) == sorted(OFFLINE_CONFIGS)


@pytest.mark.parametrize("name", BACKENDS.names())
def test_backend_answers_system_and_user(name: str) -> None:
    impl = backend(name)
    assert impl.name == name
    assert impl.model_id()
    result = impl.generate(request(Message(role="system", content="s"),
                                   Message(role="user", content=QUESTION)))  # fmt: skip
    assert isinstance(result, GenerationResult)
    assert result.ok and result.text == ANSWER and result.latency_s >= 0


@pytest.mark.parametrize("name", BACKENDS.names())
def test_backend_errors_are_results_not_exceptions(name: str) -> None:
    impl = backend(name)
    unknown = impl.generate(request(Message(role="user", content="no prepared answer")))
    assert isinstance(unknown, GenerationResult) and unknown.error
    # a request ending with the assistant answer is never sent on: it is refused
    leaked = impl.generate(request(Message(role="user", content=QUESTION),
                                   Message(role="assistant", content=ANSWER)))  # fmt: skip
    assert leaked.error and leaked.text == ""
