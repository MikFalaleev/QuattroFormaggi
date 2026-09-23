from __future__ import annotations

import importlib
import json

import pytest

from qf.common import QFError, all_registries
from qf.domain import (
    SHIPMENT_EXTRACTION,
    TASKS,
    TaskSpec,
    check_record,
    get_task,
    register_task,
    serialize_target,
)
from tests.factories import EXAMPLE_ANSWER, make_card, make_record, make_target


def test_v01_has_one_task() -> None:
    assert TASKS.names() == ["shipment_extraction"]
    task = get_task("shipment_extraction")
    assert task is SHIPMENT_EXTRACTION
    assert task.name == "shipment_extraction"
    assert task.prompts == (("card_v1", "system_extract_v1"), ("card_v2", "system_extract_v2"))
    assert task.schema_versions == ("card_v1", "card_v2")
    assert task.prompt_for("card_v2") == "system_extract_v2"
    with pytest.raises(QFError, match="does not answer in card_v9"):
        task.prompt_for("card_v9")
    assert TASKS.port is None


def test_task_registry_is_visible_after_wiring() -> None:
    importlib.import_module("qf.cli.wiring")
    assert "task" in {registry.kind for registry in all_registries()}


def test_unknown_task_name_is_explicit_error() -> None:
    with pytest.raises(QFError, match="unknown task 'no_such_task'; available: shipment_"):
        get_task("no_such_task")


def test_task_cannot_be_registered_twice() -> None:
    with pytest.raises(QFError, match="already registered"):
        register_task(SHIPMENT_EXTRACTION)


def test_task_with_unsupported_schema_is_refused() -> None:
    spec = TaskSpec(name="future_task", prompts=(("card_v9", "system_extract_v1"),))  # type: ignore[arg-type]
    with pytest.raises(QFError, match="unsupported schema version 'card_v9'"):
        register_task(spec)
    unknown_prompt = TaskSpec(name="future_task", prompts=(("card_v1", "system_extract_v9"),))
    with pytest.raises(QFError, match="unknown system prompt version"):
        register_task(unknown_prompt)
    assert "future_task" not in TASKS
    with pytest.raises(QFError, match="no answer schema"):
        TaskSpec(name="future_task", prompts=())
    with pytest.raises(QFError, match="listed twice"):
        TaskSpec(name="future_task", prompts=(("card_v1", "a"), ("card_v1", "b")))


def test_clean_record_passes() -> None:
    assert check_record(make_record()) == []


def test_unknown_task_rejected() -> None:
    issues = check_record(make_record(task="no_such_task"))
    assert [(i.record_id, i.split, i.code) for i in issues] == [
        ("qf-train-LOAD00001-0", "train", "UNKNOWN_TASK")
    ]
    assert "known: shipment_extraction" in issues[0].message


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ("", "EMPTY_ASSISTANT"),
        ("  \n", "EMPTY_ASSISTANT"),
        ("{", "TARGET_PARSE"),
        (f"```json\n{EXAMPLE_ANSWER}\n```", "TARGET_PARSE"),
        (json.dumps(json.loads(EXAMPLE_ANSWER), ensure_ascii=False, indent=2), "NOT_CANONICAL"),
        (EXAMPLE_ANSWER.replace('"value":12592', '"value":12592.0'), "NOT_CANONICAL"),
    ],
)
def test_bad_answer_is_reported(answer: str, code: str) -> None:
    assert [issue.code for issue in check_record(make_record(answer))] == [code]


def test_inconsistent_answer_is_reported() -> None:
    answer = serialize_target(make_target(make_card(origin=None), missing_fields=[]))
    issues = check_record(make_record(answer))
    assert [issue.code for issue in issues] == ["TARGET_INCONSISTENT"]
    assert "missing_fields [] != rule ['origin']" in issues[0].message
