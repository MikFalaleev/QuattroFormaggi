from __future__ import annotations

from qf.common import DataValidationError, Issue, NotImplementedStageError, QFError


def test_not_implemented_stage_message() -> None:
    exc = NotImplementedStageError("12", "train run")
    assert isinstance(exc, QFError)
    assert str(exc) == "Stage 12 is not implemented yet (command: train run)"
    assert (exc.stage, exc.command) == ("12", "train run")


def test_data_validation_error_lists_record_ids_and_truncates() -> None:
    issues = [Issue(f"rec-{i}", "train", "SCHEMA", "bad field") for i in range(7)]
    exc = DataValidationError(issues)
    message = str(exc)
    assert exc.issues == issues
    assert message.startswith("7 data validation issue(s)")
    assert "rec-0: SCHEMA: bad field" in message
    assert "rec-6" not in message
    assert "(+2 more)" in message
