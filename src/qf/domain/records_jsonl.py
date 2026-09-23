"""Parsing of SFT record files (JSONL, one record per line) — pure, without file I/O.

Shared by the data stages and the evaluation, which may not import each other. Parsing never
stops at a bad line: each line that is not a valid `SFTRecord` becomes an `Issue` (`SCHEMA`,
or `ROLE_ORDER` for messages in the wrong order).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from qf.common import Issue, format_validation_error
from qf.contracts import SFTRecord

__all__ = ["parse_sft_jsonl"]


def _line_id(line: str, fallback: str) -> str:
    try:
        data = json.loads(line)
    except ValueError:
        return fallback
    value = data.get("id") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else fallback


def parse_sft_jsonl(
    text: str, *, source: str, split: str | None = None
) -> tuple[list[SFTRecord], list[Issue]]:
    """Records of a JSONL text and one issue per line that is not a valid record; `source`
    names the file in issues of lines without a readable id."""
    records: list[SFTRecord] = []
    issues: list[Issue] = []
    for number, line in enumerate(text.splitlines(), start=1):
        try:
            records.append(SFTRecord.model_validate_json(line))
        except ValidationError as exc:
            roles = any(
                error["loc"][:1] == ("messages",) and "roles must be exactly" in error["msg"]
                for error in exc.errors()
            )
            code = "ROLE_ORDER" if roles else "SCHEMA"
            record_id = _line_id(line, f"{source}:{number}")
            issues.append(Issue(record_id, split, code, format_validation_error(exc)))
    return records, issues
