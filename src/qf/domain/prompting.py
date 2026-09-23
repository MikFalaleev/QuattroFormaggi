"""System prompt and chat messages: one implementation for data, training and runtime (step 6).

`build_messages` is the only place where the system and user messages are assembled.
"""

from __future__ import annotations

import re
from datetime import date
from functools import cache
from importlib import resources
from typing import Final

from qf.common import QFError, sha256_text
from qf.contracts import Language, Message

__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "REQUEST_DATE_LABELS",
    "build_messages",
    "load_system_prompt",
    "system_prompt_hash",
]

DEFAULT_PROMPT_VERSION: Final = "system_extract_v1"
REQUEST_DATE_LABELS: Final[dict[Language, str]] = {"ru": "Дата запроса", "en": "Request date"}
_VERSION_PATTERN: Final = re.compile(r"[a-z0-9_]+")


@cache
def load_system_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Text of `qf/domain/prompts/<version>.txt` (package data) without the final newline."""
    if not _VERSION_PATTERN.fullmatch(version):
        raise QFError(f"invalid system prompt version {version!r}")
    resource = resources.files("qf.domain").joinpath("prompts", f"{version}.txt")
    if not resource.is_file():
        raise QFError(f"unknown system prompt version '{version}'")
    return resource.read_text(encoding="utf-8").rstrip("\n")


def system_prompt_hash(version: str = DEFAULT_PROMPT_VERSION) -> str:
    return sha256_text(load_system_prompt(version))


def build_messages(
    request_text: str,
    request_date: date,
    language: Language,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[Message]:
    """[system, user]; the user message starts with the request date (plan 6.2)."""
    user = f"{REQUEST_DATE_LABELS[language]}: {request_date:%Y-%m-%d}\n\n{request_text}"
    return [
        Message(role="system", content=load_system_prompt(prompt_version)),
        Message(role="user", content=user),
    ]
