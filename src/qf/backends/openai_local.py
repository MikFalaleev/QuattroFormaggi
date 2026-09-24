"""A local OpenAI-compatible server as a generation backend (plan step 10): LM Studio or
llama-server (step 17), which differ only in `base_url`.

The client refuses non-loopback endpoints unless explicitly allowed (CLAUDE.md, constraint 6),
never falls back to another URL and never retries except once on a refused connection. The
model name is resolved from `/v1/models`, never from a file name. LM Studio also reports the
publisher, the quantization and the loaded context (`/api/v0/models/<id>`); when the config
names the expected ones, a model loaded otherwise is refused before the first request.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Final
from urllib.parse import quote, urlsplit

import httpx
from pydantic import Field

from qf.backends.registry import BACKENDS
from qf.common import QFError, StrictConfig
from qf.contracts import GenerationRequest, GenerationResult

__all__ = ["LOOPBACK_HOSTS", "LocalOpenAIClient", "ModelInfo", "OpenAILocalBackend"]

LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_TRANSPORT: httpx.BaseTransport | None = None
"""Tests replace it with an `httpx.MockTransport` (offline contract tests)."""
_ERROR_EXCERPT: Final = 300
_log = logging.getLogger(__name__)

ModelInfo = dict[str, Any]


class LocalOpenAIClient:
    """Chat completions and model listing of one local server."""

    def __init__(self, base_url: str, timeout_s: float = 120.0, *,
                 allow_non_loopback: bool = False,
                 transport: httpx.BaseTransport | None = None) -> None:  # fmt: skip
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise QFError(f"not an http(s) URL: {base_url!r}")
        if parts.hostname not in LOOPBACK_HOSTS:
            if not allow_non_loopback:
                raise QFError(f"non-local endpoint refused: {base_url} (only "
                              f"{', '.join(sorted(LOOPBACK_HOSTS))})")  # fmt: skip
            _log.warning("non-loopback endpoint allowed explicitly: %s", base_url)
        self.base_url = base_url.rstrip("/")
        self._server = f"{parts.scheme}://{parts.netloc}"
        self._http = httpx.Client(timeout=timeout_s, transport=transport or DEFAULT_TRANSPORT)

    def _get(self, url: str) -> httpx.Response:
        try:
            return self._http.get(url)
        except httpx.HTTPError as exc:
            raise QFError(f"cannot reach {url}: {type(exc).__name__}: {exc}") from exc

    def list_models(self) -> list[str]:
        response = self._get(f"{self.base_url}/models")
        if response.status_code != 200:
            raise QFError(f"GET /models: HTTP {response.status_code}")
        return [item["id"] for item in response.json().get("data", [])]

    def model_info(self, model: str) -> ModelInfo | None:
        """LM Studio's description of the model (publisher, quantization, state, loaded
        context); None when the server has no such endpoint (llama-server)."""
        response = self._get(f"{self._server}/api/v0/models/{quote(model, safe='')}")
        if response.status_code != 200:
            return None
        info: ModelInfo = response.json()
        return info

    def resolve_model(self, expected_substring: str | None) -> str:
        """The one listed model containing the substring (or the only model); an error that
        lists the models otherwise. Never guessed from a file name."""
        models = self.list_models()
        matches = [m for m in models if expected_substring is None or expected_substring in m]
        if len(matches) != 1:
            what = "no model" if not matches else f"{len(matches)} models"
            raise QFError(f"{what} match {expected_substring!r} on the server: {models}")
        return matches[0]

    def chat(self, model: str, req: GenerationRequest, *, stream: bool = True) -> GenerationResult:
        """One completion. HTTP errors, timeouts and broken streams come back in `error`;
        a refused connection is retried once; no other URL is ever tried."""
        body: dict[str, Any] = {
            "model": model, "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens, "temperature": req.temperature, "stream": stream,
        }  # fmt: skip
        if stream:
            body["stream_options"] = {"include_usage": True}
        if req.json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "answer", "schema": req.json_schema, "strict": True}}  # fmt: skip
        started = time.monotonic()
        for attempt in (1, 2):
            try:
                return self._stream(body, started) if stream else self._plain(body, started)
            except httpx.ConnectError as exc:
                if attempt == 2:
                    return _failed(started, f"connection refused: {exc}")
            except httpx.TimeoutException:
                return _failed(started, "timeout")
            except httpx.HTTPError as exc:
                return _failed(started, f"{type(exc).__name__}: {exc}")
        raise AssertionError("unreachable")

    def _plain(self, body: dict[str, Any], started: float) -> GenerationResult:
        response = self._http.post(f"{self.base_url}/chat/completions", json=body)
        if response.status_code != 200:
            return _failed(started, _http_error(response.status_code, response.text))
        data = response.json()
        usage = data.get("usage") or {}
        return GenerationResult(
            text=data["choices"][0]["message"].get("content") or "",
            latency_s=round(time.monotonic() - started, 6),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def _stream(self, body: dict[str, Any], started: float) -> GenerationResult:
        parts: list[str] = []
        ttft: float | None = None
        usage: dict[str, Any] = {}
        finished = False
        with self._http.stream("POST", f"{self.base_url}/chat/completions", json=body) as reply:
            if reply.status_code != 200:
                return _failed(started, _http_error(reply.status_code, reply.read().decode()))
            for line in reply.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    finished = True
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    return _failed(started, f"server error: {chunk['error']}")
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = round(time.monotonic() - started, 6)
                        parts.append(piece)
        latency = round(time.monotonic() - started, 6)
        if not finished:
            return GenerationResult(text="".join(parts), latency_s=latency, ttft_s=ttft,
                                    error="stream interrupted before [DONE]")  # fmt: skip
        return GenerationResult(
            text="".join(parts), latency_s=latency, ttft_s=ttft,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )  # fmt: skip


def _failed(started: float, error: str) -> GenerationResult:
    return GenerationResult(text="", latency_s=round(time.monotonic() - started, 6), error=error)


def _http_error(status: int, text: str) -> str:
    return f"HTTP {status}: {text[:_ERROR_EXCERPT]}"


@BACKENDS.register("openai_local")
class OpenAILocalBackend:
    """LM Studio (step 10) or llama-server (step 17) through the OpenAI-compatible API."""

    class Config(StrictConfig):
        base_url: str = "http://127.0.0.1:1234/v1"
        model_substring: str | None = None  # picks the model among /v1/models
        timeout_s: float = Field(default=120.0, gt=0)
        stream: bool = True  # measures the time to the first token
        allow_non_loopback: bool = False  # only for a named machine of the local network
        expected_context: int | None = Field(default=None, gt=0)  # loaded context length
        expected_quantization: str | None = None  # e.g. Q4_K_M
        runtime: str | None = None  # e.g. "LM Studio 0.4.25+1": recorded with the run

    name = "openai_local"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = LocalOpenAIClient(config.base_url, config.timeout_s,
                                        allow_non_loopback=config.allow_non_loopback)  # fmt: skip
        self._model: str | None = None
        self.info: ModelInfo | None = None

    def _resolve(self) -> str:
        if self._model is None:
            model = self.client.resolve_model(self.config.model_substring)
            info = self.client.model_info(model)
            _check_loaded(model, info, self.config)
            self._model, self.info = model, info
        return self._model

    def model_id(self) -> str:
        """`<publisher>/<id>@<quantization> ctx<context>` when the server describes the model
        (LM Studio), the plain id otherwise."""
        model = self._resolve()
        info = self.info or {}
        if not info.get("quantization"):
            return model
        publisher = f"{info['publisher']}/" if info.get("publisher") else ""
        context = info.get("loaded_context_length")
        return f"{publisher}{model}@{info['quantization']}" + (f" ctx{context}" if context else "")

    def generate(self, req: GenerationRequest) -> GenerationResult:
        if req.messages[-1].role != "user":
            return GenerationResult(text="", latency_s=0.0, error="openai_local: last message is "
                                    f"{req.messages[-1].role}, not user")  # fmt: skip
        try:
            model = self._resolve()
        except QFError as exc:
            return GenerationResult(text="", latency_s=0.0, error=str(exc))
        return self.client.chat(model, req, stream=self.config.stream)


def _check_loaded(model: str, info: ModelInfo | None, config: OpenAILocalBackend.Config) -> None:
    """A model the server would load on demand with its own settings is refused: the run must
    use the agreed context and quantization."""
    wanted = {"loaded_context_length": config.expected_context,
              "quantization": config.expected_quantization}  # fmt: skip
    if not any(wanted.values()):
        return
    if info is None:
        raise QFError(f"{model}: the server does not describe its models, so the expected "
                      "context / quantization cannot be checked")  # fmt: skip
    if info.get("state") not in (None, "loaded"):
        raise QFError(f"{model} is not loaded (state {info.get('state')!r}); load it first, e.g. "
                      f"lms load {model} --context-length {config.expected_context}")  # fmt: skip
    for key, value in wanted.items():
        if value is not None and info.get(key) != value:
            raise QFError(f"{model}: {key} is {info.get(key)!r}, the config expects {value!r}")
