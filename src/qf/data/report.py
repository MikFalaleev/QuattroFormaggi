"""Composition and length report of an SFT dataset (plan step 7).

Lengths are in characters; token lengths are added by step 11 (tokenizer audit), which passes
a token counter. Percentiles use the nearest-rank method: p = the value at position ceil(p*n).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from qf.common import (
    ArtifactRef,
    DataValidationError,
    atomic_write_text,
    read_artifact,
    start_run,
    write_artifact,
)
from qf.contracts import SFTRecord, supported_versions
from qf.data.sft_io import read_sft_records
from qf.domain import parse_target

__all__ = [
    "LENGTH_REPORT_VERSION",
    "ReportOutcome",
    "build_length_report",
    "length_report",
    "length_stats",
    "render_length_markdown",
]

LENGTH_REPORT_VERSION: Final = "length_report_v1"


def _nearest_rank(ordered: Sequence[int], p: float) -> int:
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def length_stats(lengths: Sequence[int]) -> dict[str, int]:
    """min / p50 / p95 / max by the nearest-rank method; zeros for an empty list."""
    if not lengths:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(lengths)
    return {"min": ordered[0], "p50": _nearest_rank(ordered, 0.5),
            "p95": _nearest_rank(ordered, 0.95), "max": ordered[-1]}  # fmt: skip


def _composition(records: Sequence[SFTRecord]) -> dict[str, dict[str, int]]:
    def count(values: Sequence[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    targets = [parse_target(record.messages[2].content) for record in records]
    return {
        "family": count([r.template_family for r in records]),
        "language": count([r.language for r in records]),
        "hard_case": count([",".join(r.variant.hard_cases) or "clean" for r in records]),
        "equipment_type": count([str(t.card.equipment_type) for t in targets]),
        "missing_fields": count([name for t in targets for name in t.missing_fields]),
    }


def length_report(
    datasets: Mapping[str, Sequence[SFTRecord]],
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Records per file, composition and lengths of the user and assistant messages."""
    report: dict[str, Any] = {
        "report_version": LENGTH_REPORT_VERSION,
        "records": {name: len(records) for name, records in datasets.items()},
        "composition": {name: _composition(records) for name, records in datasets.items()},
        "chars": {
            name: {role: length_stats([len(r.messages[i].content) for r in records])
                   for role, i in (("user", 1), ("assistant", 2))}
            for name, records in datasets.items()
        },
        "tokens": None,
    }  # fmt: skip
    if count_tokens is not None:
        report["tokens"] = {
            name: {role: length_stats([count_tokens(r.messages[i].content) for r in records])
                   for role, i in (("user", 1), ("assistant", 2))}
            for name, records in datasets.items()
        }  # fmt: skip
    return report


def render_length_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Dataset length report", "", "| File | Records | User chars p50 / p95 / max |"
             " Assistant chars p50 / p95 / max |", "|---|---:|---|---|"]  # fmt: skip
    for name, count in report["records"].items():
        user, assistant = report["chars"][name]["user"], report["chars"][name]["assistant"]
        lines.append(
            f"| {name} | {count} | {user['p50']} / {user['p95']} / {user['max']} | "
            f"{assistant['p50']} / {assistant['p95']} / {assistant['max']} |"
        )
    lines += ["", "Token lengths: " + ("see JSON" if report["tokens"] else "not measured yet "
              "(step 11, tokenizer audit)"), ""]  # fmt: skip
    for name, composition in report["composition"].items():
        lines += [f"## {name}", ""]
        for key, counts in composition.items():
            lines.append(f"- {key}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class ReportOutcome:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    ref: ArtifactRef
    run_dir: Path


def build_length_report(paths: Mapping[str, Path], *, root: Path, out_dir: Path) -> ReportOutcome:
    """The stage: verified dataset files -> `length_report.{json,md}` (a `metrics` artifact
    whose parents are the files) and a run manifest. Invalid records make it fail."""
    run = start_run("report", root)
    refs = {name: read_artifact(path, "sft_dataset", supported_versions("sft_dataset"), root=root)
            for name, path in paths.items()}  # fmt: skip
    datasets: dict[str, list[SFTRecord]] = {}
    for name, ref in refs.items():
        records, issues = read_sft_records(root / ref.path, name)
        if issues:
            raise DataValidationError(issues)
        datasets[name] = records
    report = length_report(datasets)
    json_path, markdown_path = out_dir / "length_report.json", out_dir / "length_report.md"
    atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(markdown_path, render_length_markdown(report))
    ref = write_artifact(json_path, "metrics", LENGTH_REPORT_VERSION, run.run_id,
                         [r.sha256 for r in refs.values()], root=root)  # fmt: skip
    run.finish(data_hashes={r.path.as_posix(): r.sha256 for r in refs.values()},
               metrics={"records": report["records"]})  # fmt: skip
    return ReportOutcome(report, json_path, markdown_path, ref, run.run_dir)
