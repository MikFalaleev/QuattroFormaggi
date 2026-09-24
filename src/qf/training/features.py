"""Training features of one SFT record (plan step 11): input ids and the loss mask.

"Variant B": the prompt is built exactly as the model sees it when used —
`apply_chat_template(messages[:2], add_generation_prompt=True)` — and the answer is appended as
its own tokens plus one EOS. Labels are -100 on the prompt, so the loss is computed on the
answer and its EOS only. The Mistral template puts the system prompt into the last user turn,
so the full dialogue [system, user, assistant] rendered by the template loses it; building the
input from that full rendering ("variant A") is therefore never done. The tokenizer comes as an
argument (any object with the methods of `ChatTokenizer`); nothing here imports transformers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from qf.common import QFError
from qf.contracts import SFTRecord

__all__ = [
    "IGNORE_INDEX",
    "ChatTokenizer",
    "Features",
    "MaskConstructionError",
    "Skip",
    "TemplateCheck",
    "build_features",
    "check_template",
    "chat_ids",
]

IGNORE_INDEX: Final = -100


class ChatTokenizer(Protocol):
    """What the features need from a tokenizer (a transformers fast tokenizer has all of it)."""

    bos_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


class MaskConstructionError(QFError):
    """The loss mask cannot be built as the plan requires: stop for a human decision."""


@dataclass(frozen=True)
class Features:
    record_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_tokens: int
    answer_tokens: int  # trainable tokens: the answer and its EOS

    def as_dict(self) -> dict[str, list[int]]:
        return {"input_ids": self.input_ids, "labels": self.labels,
                "attention_mask": [1] * len(self.input_ids)}  # fmt: skip


@dataclass(frozen=True)
class Skip:
    """A record left out of training: it is never truncated (the answer must stay whole)."""

    record_id: str
    reason: Literal["too_long"]
    tokens: int


def chat_ids(tok: ChatTokenizer, messages: Sequence[Mapping[str, str]], *,
             add_generation_prompt: bool) -> list[int]:  # fmt: skip
    """Token ids of a chat rendered by the tokenizer's template (a list or a BatchEncoding)."""
    result = tok.apply_chat_template(list(messages), tokenize=True,
                                     add_generation_prompt=add_generation_prompt)  # fmt: skip
    ids = result["input_ids"] if isinstance(result, Mapping) or hasattr(result, "keys") else result
    return [int(i) for i in ids]


def _messages(record: SFTRecord) -> list[dict[str, str]]:
    roles = [m.role for m in record.messages]
    if roles != ["system", "user", "assistant"]:
        raise MaskConstructionError(f"{record.id}: expected system, user, assistant, got {roles}")
    return [{"role": m.role, "content": m.content} for m in record.messages]


def build_features(record: SFTRecord, tok: ChatTokenizer, max_len: int) -> Features | Skip:
    """Input ids and labels of a record (variant B), or `Skip` if it does not fit `max_len`."""
    messages = _messages(record)
    prompt = chat_ids(tok, messages[:2], add_generation_prompt=True)
    if tok.eos_token_id is None:
        raise MaskConstructionError("the tokenizer has no EOS token")
    answer = [*tok.encode(messages[2]["content"], add_special_tokens=False), tok.eos_token_id]
    if len(answer) < 2:
        raise MaskConstructionError(f"{record.id}: the answer has no trainable tokens")
    if answer[-2] == tok.eos_token_id:
        raise MaskConstructionError(f"{record.id}: EOS is doubled at the end of the answer")
    input_ids = prompt + answer
    bos = tok.bos_token_id
    if bos is not None and (input_ids.count(bos) > 1 or input_ids[0] != bos):
        raise MaskConstructionError(f"{record.id}: expected one BOS at the start, found "
                                    f"{input_ids.count(bos)}")  # fmt: skip
    if len(input_ids) > max_len:
        return Skip(record.id, "too_long", len(input_ids))
    labels = [IGNORE_INDEX] * len(prompt) + answer
    return Features(record.id, input_ids, labels, len(prompt), len(answer))


@dataclass(frozen=True)
class TemplateCheck:
    """How variant B compares with the template's own rendering of the full dialogue."""

    record_id: str
    identical: bool  # the full rendering gives exactly the variant-B ids
    system_in_full: bool  # the full rendering contains the system prompt
    system_in_input: bool  # the variant-B input contains it (it must)
    answer_tokens_match: bool  # the answer tokens are the same inside the full rendering
    difference: str = ""  # what differs, decoded, when not identical
    notes: list[str] = field(default_factory=list)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_template(record: SFTRecord, tok: ChatTokenizer) -> TemplateCheck:
    """Compare variant B with `apply_chat_template(messages[:3])`. Allowed differences: the
    system prompt (a Mistral template drops it from a finished dialogue) and whitespace before
    the answer; anything else raises `MaskConstructionError` (stop for a human)."""
    messages = _messages(record)
    features = build_features(record, tok, max_len=10**9)
    assert isinstance(features, Features)
    full = chat_ids(tok, messages, add_generation_prompt=False)
    system = messages[0]["content"]
    input_text = tok.decode(features.input_ids)
    full_text = tok.decode(full)
    answer = features.input_ids[features.prompt_tokens :]
    check = TemplateCheck(
        record_id=record.id, identical=full == features.input_ids,
        system_in_full=_squash(system) in _squash(full_text),
        system_in_input=_squash(system) in _squash(input_text),
        answer_tokens_match=full[-len(answer):] == answer,
    )  # fmt: skip
    if not check.system_in_input:
        raise MaskConstructionError(f"{record.id}: the input lost the system prompt")
    if check.identical:
        return check
    without_system = _squash(input_text).replace(_squash(system), "", 1)
    if without_system != _squash(full_text) and _squash(input_text) != _squash(full_text):
        raise MaskConstructionError(f"{record.id}: variant B and the template differ beyond the "
                                    f"system prompt:\nB: {input_text[:400]}\n"
                                    f"template: {full_text[:400]}")  # fmt: skip
    difference = ("the template drops the system prompt from the finished dialogue"
                  if not check.system_in_full else "whitespace only")  # fmt: skip
    return TemplateCheck(**{**check.__dict__, "difference": difference})
