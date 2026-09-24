"""Padding of training batches (plan step 11): right padding, labels -100 on the padding.

The pad token is chosen among the tokens the tokenizer already has — the vocabulary is never
extended (CLAUDE.md, constraint 3). `torch` is imported only when tensors are asked for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qf.training.features import IGNORE_INDEX

__all__ = ["PadCollator", "choose_pad_token"]


def choose_pad_token(tok: Any) -> tuple[int, str]:
    """(pad token id, why): the tokenizer's pad token; otherwise an existing `<pad>` token;
    otherwise EOS (masked out of the loss by the labels anyway)."""
    if getattr(tok, "pad_token_id", None) is not None:
        return int(tok.pad_token_id), "the tokenizer's pad token"
    vocab = tok.get_vocab()
    if "<pad>" in vocab:
        return int(vocab["<pad>"]), "the existing <pad> token of the vocabulary"
    if tok.eos_token_id is None:
        raise ValueError("the tokenizer has neither a pad token, <pad> nor EOS")
    return int(tok.eos_token_id), "EOS (no pad token in the vocabulary)"


@dataclass(frozen=True)
class PadCollator:
    pad_token_id: int
    label_pad: int = IGNORE_INDEX
    pad_to_multiple_of: int | None = None

    def pad(self, features: Sequence[dict[str, list[int]]]) -> dict[str, list[list[int]]]:
        """The padded batch as lists (right padding)."""
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            step = self.pad_to_multiple_of
            longest = -(-longest // step) * step
        batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            gap = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * gap)
            batch["labels"].append(f["labels"] + [self.label_pad] * gap)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * gap)
        return batch

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        """The padded batch as torch tensors (for the trainer, step 12)."""
        import torch

        return {key: torch.tensor(value) for key, value in self.pad(features).items()}
