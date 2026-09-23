"""Slices of the eval report (plan step 9): each maps a benchmark record to its group."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal

from qf.contracts import SFTRecord

__all__ = ["SLICES", "SliceName", "record_kind", "slice_of"]

SliceName = Literal[
    "task", "language", "template_family", "distribution", "ood_reason", "kind", "hard_case",
    "source",
]  # fmt: skip


def record_kind(record: SFTRecord) -> str:
    """clean / missing (dropped fields only) / hard (any other hard case): the kind slice that
    the step 10 gates read, e.g. hallucinated required fields on hard cases."""
    hard_cases = record.variant.hard_cases
    if not hard_cases:
        return "clean"
    return "missing" if hard_cases == ["dropped_fields"] else "hard"


SLICES: Final[dict[str, Callable[[SFTRecord], str]]] = {
    "task": lambda r: r.task,
    "language": lambda r: r.language,
    "template_family": lambda r: r.template_family,
    "distribution": lambda r: "ood" if r.variant.ood_reason else "in_dist",
    "ood_reason": lambda r: r.variant.ood_reason or "—",
    "kind": record_kind,
    "hard_case": lambda r: ",".join(r.variant.hard_cases) or "clean",
    "source": lambda r: "synthetic" if r.synthetic else "manual",
}


def slice_of(record: SFTRecord, name: str) -> str:
    return SLICES[name](record)
