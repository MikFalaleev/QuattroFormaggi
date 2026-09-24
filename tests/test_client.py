"""The local OpenAI-compatible client and backend of step 10 (LM Studio), offline with
`httpx.MockTransport`; one live request is marked `lmstudio`."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from qf.backends import LocalOpenAIClient, OpenAILocalBackend
from qf.backends import openai_local as module
from qf.common import QFError
from qf.contracts import GenerationRequest, Message

MODELS = {"data": [{"id": "mistral-nemo-instruct-2407"}, {"id": "qwen3-1.7b"}]}
INFO = {"id": "mistral-nemo-instruct-2407", "publisher": "lmstudio-community",
        "quantization": "Q4_K_M", "state": "loaded", "loaded_context_length": 4096}  # fmt: skip
ANSWER = '{"card": {}}'
Handler = Callable[[httpx.Request], httpx.Response]


def sse(*pieces: str, done: bool = True, usage: bool = True) -> str:
    chunks = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
    if usage:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 3}})
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return body + ("data: [DONE]\n\n" if done else "")


def server(seen: list[httpx.Request] | None = None, *, chat: Handler | None = None,
           info: dict[str, Any] | None = INFO) -> httpx.MockTransport:  # fmt: skip
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        if path.startswith("/api/v0/models/"):
            return httpx.Response(200, json=info) if info else httpx.Response(404)
        if path == "/v1/chat/completions":
            if chat is not None:
                return chat(request)
            return httpx.Response(200, text=sse('{"card"', ": {}}"),
                                  headers={"content-type": "text/event-stream"})  # fmt: skip
        return httpx.Response(404)

    return httpx.MockTransport(handle)


def request(json_schema: dict[str, Any] | None = None) -> GenerationRequest:
    return GenerationRequest(messages=[Message(role="system", content="s"),
                                       Message(role="user", content="заявка")],
                             max_tokens=64, temperature=0.0, json_schema=json_schema)  # fmt: skip


def client(transport: httpx.MockTransport) -> LocalOpenAIClient:
    return LocalOpenAIClient("http://127.0.0.1:1234/v1", transport=transport)


def test_refuses_non_loopback_url() -> None:
    for url in ("http://10.0.0.5:1234/v1", "https://api.example.com/v1", "ftp://127.0.0.1/"):
        with pytest.raises(QFError):
            LocalOpenAIClient(url)
    allowed = LocalOpenAIClient("http://10.0.0.5:1234/v1", allow_non_loopback=True)
    assert allowed.base_url == "http://10.0.0.5:1234/v1"
    assert LocalOpenAIClient("http://localhost:1234/v1/").base_url == "http://localhost:1234/v1"


def test_resolve_model_from_api_not_filename() -> None:
    api = client(server())
    assert api.resolve_model("mistral-nemo") == "mistral-nemo-instruct-2407"
    assert api.model_info("mistral-nemo-instruct-2407") == INFO
    assert client(server(info=None)).model_info("x") is None


def test_resolve_model_ambiguous_raises() -> None:
    api = client(server())
    with pytest.raises(QFError, match="2 models"):
        api.resolve_model(None)
    with pytest.raises(QFError, match="no model"):
        api.resolve_model("llama")


def test_stream_ttft_measured() -> None:
    result = client(server()).chat("m", request())
    assert result.ok and result.text == ANSWER
    assert result.ttft_s is not None and result.ttft_s <= result.latency_s
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)


def test_plain_request_and_http_error() -> None:
    def plain(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ANSWER}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    result = client(server(chat=plain)).chat("m", request(), stream=False)
    assert result.text == ANSWER and result.prompt_tokens == 5 and result.ttft_s is None
    failing = client(server(chat=lambda r: httpx.Response(500, text="boom")))
    for stream in (True, False):
        error = failing.chat("m", request(), stream=stream)
        assert error.error == "HTTP 500: boom" and error.text == ""


def test_json_schema_payload_shape() -> None:
    seen: list[httpx.Request] = []
    schema = {"type": "object", "properties": {"card": {"type": "object"}}}
    client(server(seen)).chat("mistral", request(json_schema=schema))
    body = json.loads(seen[-1].content)
    assert body["model"] == "mistral" and body["temperature"] == 0.0 and body["max_tokens"] == 64
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": schema, "strict": True}}  # fmt: skip
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    client(server(seen)).chat("mistral", request())
    assert "response_format" not in json.loads(seen[-1].content)


def test_timeout_returns_error_result() -> None:
    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    result = client(server(chat=slow)).chat("m", request())
    assert result.error == "timeout" and result.text == ""


def test_broken_stream_and_server_error() -> None:
    cut = client(server(chat=lambda r: httpx.Response(200, text=sse("{", done=False))))
    result = cut.chat("m", request())
    assert result.error == "stream interrupted before [DONE]" and result.text == "{"
    bad = client(server(chat=lambda r: httpx.Response(
        200, text='data: {"error": "model crashed"}\n\n')))  # fmt: skip
    assert bad.chat("m", request()).error == "server error: model crashed"
    other = client(server(chat=lambda r: (_ for _ in ()).throw(httpx.RemoteProtocolError("x"))))
    assert other.chat("m", request()).error == "RemoteProtocolError: x"


def test_no_fallback_on_failure() -> None:
    seen: list[httpx.Request] = []

    def refused(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        raise httpx.ConnectError("refused", request=req)

    result = client(httpx.MockTransport(refused)).chat("m", request())
    assert result.error and result.error.startswith("connection refused")
    assert len(seen) == 2  # one retry, same URL, nothing else
    assert {str(r.url) for r in seen} == {"http://127.0.0.1:1234/v1/chat/completions"}


def test_unreachable_server_is_an_error() -> None:
    def refused(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(QFError, match="cannot reach"):
        client(httpx.MockTransport(refused)).list_models()
    broken = client(httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(QFError, match="HTTP 503"):
        broken.list_models()


def backend(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport,
            **config: Any) -> OpenAILocalBackend:  # fmt: skip
    monkeypatch.setattr(module, "DEFAULT_TRANSPORT", transport)
    return OpenAILocalBackend(OpenAILocalBackend.Config(model_substring="mistral", **config))


def test_backend_model_id_and_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = backend(monkeypatch, server(), expected_context=4096, expected_quantization="Q4_K_M")
    assert ok.model_id() == "lmstudio-community/mistral-nemo-instruct-2407@Q4_K_M ctx4096"
    assert ok.generate(request()).text == ANSWER
    wrong = backend(monkeypatch, server(), expected_context=8192)
    with pytest.raises(QFError, match="loaded_context_length is 4096"):
        wrong.model_id()
    assert "8192" in (wrong.generate(request()).error or "")
    unloaded = backend(monkeypatch, server(info={**INFO, "state": "not-loaded"}),
                       expected_quantization="Q4_K_M")  # fmt: skip
    with pytest.raises(QFError, match="not loaded"):
        unloaded.model_id()
    llama = backend(monkeypatch, server(info=None))
    assert llama.model_id() == "mistral-nemo-instruct-2407"
    with pytest.raises(QFError, match="does not describe"):
        backend(monkeypatch, server(info=None), expected_context=4096).model_id()
    leaked = ok.generate(GenerationRequest(messages=[Message(role="user", content="x"),
                                                     Message(role="assistant", content="y")],
                                           max_tokens=8, temperature=0.0))  # fmt: skip
    assert leaked.error and "not user" in leaked.error


@pytest.mark.lmstudio
def test_lmstudio_live() -> None:
    """One real request to the model loaded in LM Studio (step 10 setup)."""
    live = OpenAILocalBackend(OpenAILocalBackend.Config(
        model_substring="mistral-nemo-instruct-2407", expected_context=4096,
        expected_quantization="Q4_K_M"))  # fmt: skip
    assert live.model_id().endswith("@Q4_K_M ctx4096")
    result = live.generate(GenerationRequest(
        messages=[Message(role="user", content="Ответь одним словом: да")], max_tokens=8,
        temperature=0.0))  # fmt: skip
    assert result.ok and result.text.strip() and result.ttft_s is not None
