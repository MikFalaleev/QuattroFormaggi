"""Types exchanged with any text-generation backend (port `GenerationBackend`, plan C.4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["GenerationRequest", "GenerationResult", "Message", "Role"]

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """One chat message. Declared once here; SFT records (step 4) reuse it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    json_schema: dict[str, Any] | None = None


class GenerationResult(BaseModel):
    """Backend output. Failures are recorded in `error`, never raised or masked (plan step 9)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    latency_s: float = Field(ge=0.0)
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
