"""A backend with prepared answers, for tests and harness checks (plan step 9). Not a model.

Answers are looked up by the sha256 of the last (user) message. The special answer
`__timeout__` simulates a timeout; a request without a prepared answer is an error result.
The model id carries a hash of the answers, so a run records which answers it was given and
cannot be resumed with other ones.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import Field

from qf.backends.registry import BACKENDS
from qf.common import QFError, StrictConfig, project_root, sha256_json, sha256_text
from qf.contracts import GenerationRequest, GenerationResult

__all__ = ["TIMEOUT", "FakeBackend"]

TIMEOUT: Final = "__timeout__"


@BACKENDS.register("fake")
class FakeBackend:
    """Returns prepared answers; checks that the request ends with a user message."""

    class Config(StrictConfig):
        responses: dict[str, str] = Field(default_factory=dict)  # sha256(user message) -> text
        responses_file: Path | None = None  # JSON object of the same kind, relative to the root
        model: str = "fake"

    name = "fake"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.responses: dict[str, str] = dict(config.responses)
        if config.responses_file is not None:
            path = config.responses_file
            path = path if path.is_absolute() else project_root() / path
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise QFError(f"cannot read fake responses {path}: {exc}") from exc
            if not isinstance(loaded, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
            ):
                raise QFError(f"{path}: expected a JSON object of sha256 -> answer text")
            self.responses |= loaded
        self.requests: list[GenerationRequest] = []

    @classmethod
    def from_answers(cls, answers: Mapping[str, str], model: str = "fake") -> FakeBackend:
        """A backend answering `answers[user_message]` (keys are the plain user messages)."""
        hashed = {sha256_text(text): answer for text, answer in answers.items()}
        return cls(cls.Config(responses=hashed, model=model))

    def model_id(self) -> str:
        """`fake:<model>@<sha256 of the answers, 12 hex>`: which answers produced a run."""
        return f"fake:{self.config.model}@{sha256_json(self.responses)[:12]}"

    def generate(self, req: GenerationRequest) -> GenerationResult:
        started = time.monotonic()
        self.requests.append(req)
        last = req.messages[-1]
        if last.role != "user":
            error = f"fake backend: last message is {last.role}, not user"
            return GenerationResult(text="", latency_s=0.0, error=error)
        answer = self.responses.get(sha256_text(last.content))
        elapsed = round(time.monotonic() - started, 6)
        if answer is None:
            return GenerationResult(text="", latency_s=elapsed, error="fake backend: no answer")
        if answer == TIMEOUT:
            return GenerationResult(text="", latency_s=elapsed, error="timeout")
        return GenerationResult(text=answer, latency_s=elapsed)
