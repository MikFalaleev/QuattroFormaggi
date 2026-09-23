"""Registry of the model's tasks (D-040): the set of tasks is open.

A task binds a system prompt to an answer schema. A new task (roadmap v0.2) is a new
`register_task(...)` call; the record contract (`SFTRecord.task: TaskName`) does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from qf.common import Registry
from qf.contracts import CARD_SCHEMA_VERSION, TargetSchemaVersion, require_supported

__all__ = ["SHIPMENT_EXTRACTION", "TASKS", "TaskSpec", "get_task", "register_task"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    system_prompt_version: str  # prompt file stem in qf/domain/prompts/, e.g. "system_extract_v1"
    target_schema_version: TargetSchemaVersion


TASKS: Registry[TaskSpec] = Registry("task")


def register_task(spec: TaskSpec) -> TaskSpec:
    require_supported("target", spec.target_schema_version)
    TASKS.register(spec.name)(spec)
    return spec


def get_task(name: str) -> TaskSpec:
    """The registered task; QFError listing the known tasks otherwise."""
    return TASKS.get(name)


SHIPMENT_EXTRACTION = register_task(
    TaskSpec(
        name="shipment_extraction",
        system_prompt_version="system_extract_v1",
        target_schema_version=CARD_SCHEMA_VERSION,
    )
)
