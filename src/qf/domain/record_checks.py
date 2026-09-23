"""Checks of one SFT record on its own (plan step 4).

`qf validate-data` (step 7) runs this for every record and adds the cross-record checks
(leaks, duplicates, splits). Issue codes are the ones of step 7.
"""

from __future__ import annotations

from qf.common import Issue
from qf.contracts import SFTRecord
from qf.domain.rules import check_target_consistency
from qf.domain.serialization import TargetParseError, parse_target, serialize_target
from qf.domain.tasks import TASKS

__all__ = ["check_record"]


def check_record(record: SFTRecord) -> list[Issue]:
    """Problems of a record that already validates as `SFTRecord`; empty if it is clean.

    Codes: UNKNOWN_TASK, EMPTY_ASSISTANT, TARGET_PARSE, NOT_CANONICAL, TARGET_INCONSISTENT.
    The answer is parsed as card_v1, the only schema of `TargetSchemaVersion` so far.
    """

    def issue(code: str, message: str) -> Issue:
        return Issue(record_id=record.id, split=record.split, code=code, message=message)

    if record.task not in TASKS:
        known = ", ".join(TASKS.names())
        return [issue("UNKNOWN_TASK", f"unknown task '{record.task}'; known: {known}")]
    answer = record.messages[2].content
    if not answer.strip():
        return [issue("EMPTY_ASSISTANT", "the assistant message is empty")]
    try:
        target = parse_target(answer)
    except TargetParseError as exc:
        return [issue("TARGET_PARSE", str(exc))]
    issues = []
    if serialize_target(target) != answer:
        issues.append(issue("NOT_CANONICAL", "the answer differs from serialize_target(answer)"))
    issues += [issue("TARGET_INCONSISTENT", p) for p in check_target_consistency(target)]
    return issues
