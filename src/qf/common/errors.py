"""Exception hierarchy shared by every qf package."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["DataValidationError", "Issue", "NotImplementedStageError", "QFError"]

_MAX_ISSUES_IN_MESSAGE = 5


class QFError(Exception):
    """Expected, user-facing error. The CLI turns it into exit code 1."""


class NotImplementedStageError(QFError):
    """Raised by commands whose plan step is not implemented yet. The CLI returns exit code 2."""

    def __init__(self, stage: str, command: str) -> None:
        self.stage = stage
        self.command = command
        super().__init__(f"Stage {stage} is not implemented yet (command: {command})")


@dataclass(frozen=True)
class Issue:
    """One problem found in a data record: which record, where, what kind and why."""

    record_id: str
    split: str | None
    code: str
    message: str


class DataValidationError(QFError):
    """Raised when data records fail validation; carries every issue found."""

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues = list(issues)
        shown = "; ".join(
            f"{issue.record_id}: {issue.code}: {issue.message}"
            for issue in self.issues[:_MAX_ISSUES_IN_MESSAGE]
        )
        more = len(self.issues) - _MAX_ISSUES_IN_MESSAGE
        suffix = f" (+{more} more)" if more > 0 else ""
        super().__init__(f"{len(self.issues)} data validation issue(s): {shown}{suffix}")
