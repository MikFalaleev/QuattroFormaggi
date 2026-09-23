"""Registry of the model's tasks (D-040): the set of tasks is open.

A task binds each answer schema it supports to a system prompt (D-086): `shipment_extraction`
answers card_v1 with `system_extract_v1` and card_v2 with `system_extract_v2`, so records of
both versions stay valid. A new task (roadmap v0.2) is a new `register_task(...)` call; the
record contract (`SFTRecord.task: TaskName`) does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from qf.common import QFError, Registry
from qf.contracts import (
    CARD_SCHEMA_VERSION,
    CARD_V2_SCHEMA_VERSION,
    TargetSchemaVersion,
    require_supported,
)
from qf.domain.prompting import load_system_prompt

__all__ = ["SHIPMENT_EXTRACTION", "TASKS", "TaskSpec", "get_task", "register_task"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompts: tuple[tuple[TargetSchemaVersion, str], ...]
    """(answer schema, system prompt version) pairs, oldest first; a prompt version is a file
    stem in qf/domain/prompts/, e.g. "system_extract_v1"."""

    def __post_init__(self) -> None:
        if not self.prompts:
            raise QFError(f"task {self.name}: no answer schema")
        if len({schema for schema, _ in self.prompts}) != len(self.prompts):
            raise QFError(f"task {self.name}: an answer schema is listed twice")

    @property
    def schema_versions(self) -> tuple[TargetSchemaVersion, ...]:
        return tuple(schema for schema, _ in self.prompts)

    def prompt_for(self, schema_version: str) -> str:
        """The system prompt version of the schema; QFError if the task does not answer it."""
        for schema, prompt in self.prompts:
            if schema == schema_version:
                return prompt
        known = ", ".join(self.schema_versions)
        raise QFError(f"task {self.name} does not answer in {schema_version} (answers: {known})")


TASKS: Registry[TaskSpec] = Registry("task")


def register_task(spec: TaskSpec) -> TaskSpec:
    for schema, prompt in spec.prompts:
        require_supported("target", schema)
        load_system_prompt(prompt)  # an unknown prompt fails at registration
    TASKS.register(spec.name)(spec)
    return spec


def get_task(name: str) -> TaskSpec:
    """The registered task; QFError listing the known tasks otherwise."""
    return TASKS.get(name)


SHIPMENT_EXTRACTION = register_task(
    TaskSpec(
        name="shipment_extraction",
        prompts=(
            (CARD_SCHEMA_VERSION, "system_extract_v1"),
            (CARD_V2_SCHEMA_VERSION, "system_extract_v2"),
        ),
    )
)
