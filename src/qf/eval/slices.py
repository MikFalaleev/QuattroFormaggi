"""Slices of the eval report (plan step 9): each maps a benchmark record to its group."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal

from qf.contracts import SFTRecord
from qf.domain import get_target_schema

__all__ = ["MANUAL_FAMILIES", "SLICES", "SliceName", "record_kind", "slice_of"]

MANUAL_FAMILIES: Final = ("manual", "manual_mock")
"""Template families of hand-written cases (a person's or the agent's mocks, D-103): they carry
no hard cases in `variant`, so their kind is `manual`, never `clean`."""

SliceName = Literal[
    "task", "language", "template_family", "distribution", "ood_reason", "kind", "hard_case",
    "source",
]  # fmt: skip


def _has_conditions(record: SFTRecord) -> bool:
    card = get_target_schema(record.schema_version).parse(record.messages[2].content).card
    return bool(getattr(card, "special_conditions", ()))


def record_kind(record: SFTRecord) -> str:
    """clean / conditions (card_v2: special conditions, no hard case) / missing (dropped fields
    only) / hard (any other hard case) / manual (hand-written cases): the kind slice that the
    step 10 gates read, e.g. hallucinated required fields on hard cases."""
    if record.template_family in MANUAL_FAMILIES:
        return "manual"
    hard_cases = record.variant.hard_cases
    if not hard_cases:
        return "conditions" if _has_conditions(record) else "clean"
    return "missing" if hard_cases == ["dropped_fields"] else "hard"


def _hard_case(record: SFTRecord) -> str:
    if record.template_family in MANUAL_FAMILIES:
        return "manual"
    return ",".join(record.variant.hard_cases) or "clean"


SLICES: Final[dict[str, Callable[[SFTRecord], str]]] = {
    "task": lambda r: r.task,
    "language": lambda r: r.language,
    "template_family": lambda r: r.template_family,
    "distribution": lambda r: "ood" if r.variant.ood_reason else "in_dist",
    "ood_reason": lambda r: r.variant.ood_reason or "—",
    "kind": record_kind,
    "hard_case": _hard_case,
    "source": lambda r: "synthetic" if r.synthetic else "manual",
}


def slice_of(record: SFTRecord, name: str) -> str:
    return SLICES[name](record)
