"""Reading and writing SFT record files (JSONL, one record per line) as artifacts.

Parsing is `qf.domain.parse_sft_jsonl` (shared with the evaluation): a bad line becomes an
`Issue`, so a validator can report all of them at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from qf.common import (
    ArtifactKind,
    ArtifactRef,
    Issue,
    atomic_write_text,
    write_artifact,
)
from qf.contracts import SFT_RECORD_SCHEMA_VERSION, SFTRecord
from qf.domain import parse_sft_jsonl

__all__ = ["DATASET_FILES", "SPLIT_FILES", "read_sft_records", "write_sft_records"]

SPLIT_FILES: Final = ("train", "val", "test", "test_ood")
DATASET_FILES: Final = (*SPLIT_FILES, "smoke")  # smoke: the first train records, not a split


def read_sft_records(path: Path, split: str | None = None) -> tuple[list[SFTRecord], list[Issue]]:
    """Records of a JSONL file and one issue per line that is not a valid record."""
    return parse_sft_jsonl(path.read_text(encoding="utf-8"), source=path.name, split=split)


def write_sft_records(
    records: Sequence[SFTRecord],
    path: Path,
    *,
    run_id: str,
    parents: Sequence[str],
    root: Path,
    kind: ArtifactKind = "sft_dataset",
    extra: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Write the records (in the given order) and the artifact manifest next to them."""
    atomic_write_text(path, "".join(record.model_dump_json() + "\n" for record in records))
    return write_artifact(path, kind, SFT_RECORD_SCHEMA_VERSION, run_id, parents, extra=extra,
                          root=root)  # fmt: skip
