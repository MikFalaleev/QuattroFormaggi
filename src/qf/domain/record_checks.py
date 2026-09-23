"""Checks of one SFT record on its own (plan step 4).

`qf validate-data` (step 7) runs this for every record and adds the cross-record checks
(leaks, duplicates, splits). Issue codes are the ones of step 7.
"""

from __future__ import annotations

from qf.common import Issue
from qf.contracts import SFTRecord
from qf.domain.schemas import get_target_schema
from qf.domain.serialization import TargetParseError
from qf.domain.tasks import TASKS

__all__ = ["check_record"]


def check_record(record: SFTRecord) -> list[Issue]:
    """Problems of a record that already validates as `SFTRecord`; empty if it is clean.

    Codes: UNKNOWN_TASK, UNSUPPORTED_SCHEMA (the task does not answer in the record's
    `schema_version`), EMPTY_ASSISTANT, TARGET_PARSE, NOT_CANONICAL, TARGET_INCONSISTENT. The
    answer is checked with the rules of its schema version (`TARGET_SCHEMAS`, D-086).
    """

    def issue(code: str, message: str) -> Issue:
        return Issue(record_id=record.id, split=record.split, code=code, message=message)

    if record.task not in TASKS:
        known = ", ".join(TASKS.names())
        return [issue("UNKNOWN_TASK", f"unknown task '{record.task}'; known: {known}")]
    task = TASKS.get(record.task)
    if record.schema_version not in task.schema_versions:
        answers = ", ".join(task.schema_versions)
        message = f"task {task.name} answers in {answers}, not {record.schema_version}"
        return [issue("UNSUPPORTED_SCHEMA", message)]
    schema = get_target_schema(record.schema_version)
    answer = record.messages[2].content
    if not answer.strip():
        return [issue("EMPTY_ASSISTANT", "the assistant message is empty")]
    try:
        target = schema.parse(answer)
    except TargetParseError as exc:
        return [issue("TARGET_PARSE", str(exc))]
    issues = []
    if schema.serialize(target) != answer:
        issues.append(issue("NOT_CANONICAL", "the answer differs from serialize_target(answer)"))
    issues += [issue("TARGET_INCONSISTENT", p) for p in schema.check_consistency(target)]
    return issues
