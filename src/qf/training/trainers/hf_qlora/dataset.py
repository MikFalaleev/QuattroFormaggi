"""Training features as a torch dataset (plan step 12).

Each item carries `record_index`, the position of its record in the file: the collator turns it
into a tensor and the trainer removes it before the forward pass, logging which records every
optimizer step saw (the data position that resume must restore).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from qf.common import QFError
from qf.training import Features, PadCollator, Skip, build_features, load_records

__all__ = ["RECORD_INDEX", "FeatureDataset", "IndexedCollator", "load_features"]

RECORD_INDEX = "record_index"


class FeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, features: Sequence[Features]) -> None:
        self.features = list(features)
        self.record_ids = [f.record_id for f in self.features]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.features[index].as_dict(), RECORD_INDEX: index}

    @property
    def tokens_total(self) -> int:
        return sum(len(f.input_ids) for f in self.features)


def load_features(path: Path, tok: Any, max_len: int, expected_sha256: str, *,
                  max_skipped_share: float) -> tuple[FeatureDataset, list[Skip]]:  # fmt: skip
    """Variant-B features of the records at `path` (sha256 checked before reading). Records
    longer than `max_len` are skipped, never truncated; more than `max_skipped_share` of them
    stops the run."""
    records = load_records(path, expected_sha256)
    built = [build_features(record, tok, max_len) for record in records]
    skipped = [b for b in built if isinstance(b, Skip)]
    if not records or len(skipped) / len(records) > max_skipped_share:
        raise QFError(f"{path}: {len(skipped)} of {len(records)} records are longer than "
                      f"{max_len} tokens (allowed share {max_skipped_share:.0%})")  # fmt: skip
    return FeatureDataset([b for b in built if isinstance(b, Features)]), skipped


@dataclass(frozen=True)
class IndexedCollator:
    pad: PadCollator

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.pad([{k: v for k, v in item.items() if k != RECORD_INDEX} for item in items])
        batch[RECORD_INDEX] = torch.tensor([item[RECORD_INDEX] for item in items])
        return batch
