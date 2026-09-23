"""Reading and writing SFT record files (JSONL, one record per line) as artifacts.

Reading never stops at a bad line: every line that is not a valid `SFTRecord` becomes an
`Issue` (`SCHEMA`, or `ROLE_ORDER` for messages in the wrong order), so a validator can report
all of them at once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from qf.common import (
    ArtifactKind,
    ArtifactRef,
    Issue,
    atomic_write_text,
    format_validation_error,
    write_artifact,
)
from qf.contracts import SFT_RECORD_SCHEMA_VERSION, SFTRecord

__all__ = ["DATASET_FILES", "SPLIT_FILES", "read_sft_records", "write_sft_records"]

SPLIT_FILES: Final = ("train", "val", "test", "test_ood")
DATASET_FILES: Final = (*SPLIT_FILES, "smoke")  # smoke: the first train records, not a split


def _line_id(line: str, fallback: str) -> str:
    try:
        data = json.loads(line)
    except ValueError:
        return fallback
    value = data.get("id") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else fallback


def read_sft_records(path: Path, split: str | None = None) -> tuple[list[SFTRecord], list[Issue]]:
    """Records of a JSONL file and one issue per line that is not a valid record."""
    records: list[SFTRecord] = []
    issues: list[Issue] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(SFTRecord.model_validate_json(line))
        except ValidationError as exc:
            roles = any(
                error["loc"][:1] == ("messages",) and "roles must be exactly" in error["msg"]
                for error in exc.errors()
            )
            code = "ROLE_ORDER" if roles else "SCHEMA"
            record_id = _line_id(line, f"{path.name}:{number}")
            issues.append(Issue(record_id, split, code, format_validation_error(exc)))
    return records, issues


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
