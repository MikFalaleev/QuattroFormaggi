"""Contract test of every registered generation backend (plan C.8, step 9).

Each backend must be testable offline: `OFFLINE_CONFIGS` holds a configuration that needs no
network, GPU or model (`openai_local`, step 10: a mock LM Studio; `hf_local`, step 14: a tiny
random model on the CPU, skipped without torch). A backend registered without an entry here
fails `test_every_backend_has_an_offline_config`. Backends that answer from prepared answers
(`fake`, the mock server) return exactly `ANSWER`; the lower-bound baselines (D-109) compute a
valid card_v2 answer for any request; a model (`hf_local`) answers with its own text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import qf.cli.wiring  # noqa: F401  (every registry and lazy entry)
from qf.backends import BACKENDS
from qf.backends import openai_local as openai_local_module
from qf.cli.wiring import build
from qf.common import ComponentConfig, sha256_text
from qf.contracts import GenerationBackend, GenerationRequest, GenerationResult, Message
from qf.domain import get_target_schema

QUESTION = "Нужна машина из Перми в Самару"
ANSWER = '{"card": {}}'
OFFLINE_CONFIGS: dict[str, dict[str, Any]] = {
    "fake": {"responses": {sha256_text(QUESTION): ANSWER}},
    "openai_local": {"model_substring": "mistral", "expected_context": 4096},
    "baseline_empty": {},
    "baseline_rules": {},
    "hf_local": {},  # filled from `tests.tiny_hf.hf_config()` in a fake project root
}
COMPUTES_ANSWERS = {"baseline_empty", "baseline_rules"}
MODELS = {"hf_local"}  # answer any request with generated text


def _mock_lmstudio(request: httpx.Request) -> httpx.Response:
    """Answers QUESTION with ANSWER and anything else with HTTP 404, like a prepared server."""
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "mistral-nemo-instruct-2407"}]})
    if request.url.path.startswith("/api/v0/models/"):
        return httpx.Response(200, json={"state": "loaded", "loaded_context_length": 4096,
                                         "quantization": "Q4_K_M"})  # fmt: skip
    body = json.loads(request.content)
    if body["messages"][-1]["content"] != QUESTION:
        return httpx.Response(404, text="no prepared answer")
    chunk = json.dumps({"choices": [{"delta": {"content": ANSWER}}]})
    return httpx.Response(200, text=f"data: {chunk}\n\ndata: [DONE]\n\n")


@pytest.fixture(autouse=True)
def offline_lmstudio(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(openai_local_module, "DEFAULT_TRANSPORT",
                        httpx.MockTransport(_mock_lmstudio))  # fmt: skip
    yield


def request(*messages: Message) -> GenerationRequest:
    return GenerationRequest(messages=list(messages), max_tokens=64, temperature=0.0)


@pytest.fixture
def model_root(request: pytest.FixtureRequest, tmp_path: Path,
               monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """A fake project root with the tiny base model, for backends that load a model."""
    if request.node.callspec.params.get("name") not in MODELS:
        return
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tests.tiny_hf import tiny_base

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "quattro-formaggi"\n',
                                             encoding="utf-8")  # fmt: skip
    monkeypatch.setenv("QF_PROJECT_ROOT", str(tmp_path))
    tiny_base(tmp_path)


def backend(name: str) -> GenerationBackend:
    params = OFFLINE_CONFIGS[name]
    if name in MODELS:
        from tests.tiny_hf import hf_config

        params = hf_config()
    instance = build(BACKENDS, ComponentConfig(name=name, **params))
    assert isinstance(instance, GenerationBackend)
    return instance


def test_every_backend_has_an_offline_config() -> None:
    assert sorted(BACKENDS.names()) == sorted(OFFLINE_CONFIGS)


@pytest.mark.parametrize("name", BACKENDS.names())
@pytest.mark.usefixtures("model_root")
def test_backend_answers_system_and_user(name: str) -> None:
    impl = backend(name)
    assert impl.name == name
    assert impl.model_id()
    result = impl.generate(request(Message(role="system", content="s"),
                                   Message(role="user", content=QUESTION)))  # fmt: skip
    assert isinstance(result, GenerationResult)
    assert result.ok and result.latency_s >= 0
    if name in COMPUTES_ANSWERS:
        get_target_schema("card_v2").parse(result.text)  # a strictly valid answer
    elif name in MODELS:
        assert isinstance(result.text, str) and result.completion_tokens
    else:
        assert result.text == ANSWER


@pytest.mark.parametrize("name", BACKENDS.names())
@pytest.mark.usefixtures("model_root")
def test_backend_errors_are_results_not_exceptions(name: str) -> None:
    impl = backend(name)
    if name not in COMPUTES_ANSWERS | MODELS:  # a baseline or a model answers any request
        unknown = impl.generate(request(Message(role="user", content="no prepared answer")))
        assert isinstance(unknown, GenerationResult) and unknown.error
    # a request ending with the assistant answer is never sent on: it is refused
    leaked = impl.generate(request(Message(role="user", content=QUESTION),
                                   Message(role="assistant", content=ANSWER)))  # fmt: skip
    assert leaked.error and leaked.text == ""
