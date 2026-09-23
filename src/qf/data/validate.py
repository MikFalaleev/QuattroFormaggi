"""Independent validation of an SFT dataset (plan step 7): schema, leaks, duplicates.

The validator reads only artifacts and contracts, never the generator, so it checks synthetic,
hand-written and future real records alike. Every problem is an `Issue` with the record id,
split, code and reason; duplicates inside one split are warnings, everything else is an error.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from qf.common import (
    ArtifactRef,
    Issue,
    QFError,
    atomic_write_text,
    lineage,
    load_artifact_manifest,
    read_artifact,
    start_run,
    write_artifact,
)
from qf.contracts import SFTRecord, supported_versions
from qf.data.facts import load_facts
from qf.data.registries import TEMPLATE_FAMILIES
from qf.data.sft_io import SPLIT_FILES, read_sft_records
from qf.data.split import split_issues
from qf.domain import REQUEST_DATE_LABELS, TASKS, check_record, get_task, load_system_prompt

__all__ = [
    "MAIN_SPLITS",
    "VALIDATION_REPORT_VERSION",
    "IssueRecord",
    "ValidationOutcome",
    "ValidationReport",
    "normalized_request",
    "run_validation",
    "validate_dataset",
]

VALIDATION_REPORT_VERSION: Final = "validation_report_v1"
MAIN_SPLITS: Final = ("train", "val", "test")  # held-out routes and families must not be here
_WARNING_CODES: Final = frozenset({"DUP_EXACT_IN_SPLIT"})


class IssueRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    split: str | None
    code: str
    message: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: Literal["validation_report_v1"] = VALIDATION_REPORT_VERSION
    datasets: dict[str, dict[str, Any]]  # file name -> path, sha256, records
    bench: dict[str, Any] | None
    counts: dict[str, int]  # issue code -> number of issues
    errors: list[IssueRecord]
    warnings: list[IssueRecord]

    @property
    def passed(self) -> bool:
        return not self.errors


def normalized_request(record: SFTRecord) -> str:
    """User text without the request-date header, lower-cased, whitespace collapsed (any
    Unicode space, so a plain and a non-breaking thousands separator are equal)."""
    text = record.messages[1].content
    head, separator, body = text.partition("\n\n")
    if separator and any(head.startswith(f"{label}:") for label in REQUEST_DATE_LABELS.values()):
        text = body
    return " ".join(text.lower().split())


@cache
def _prompt(version: str) -> str:
    return load_system_prompt(version)


def _record_issues(name: str, records: Sequence[SFTRecord]) -> list[Issue]:
    """check_record (task, answer parse, canonical form, consistency) plus PROMPT_MISMATCH."""
    issues: list[Issue] = []
    for record in records:
        issues += check_record(record)
        if record.task in TASKS:
            version = get_task(record.task).system_prompt_version
            if record.messages[0].content != _prompt(version):
                issues.append(Issue(record.id, name, "PROMPT_MISMATCH",
                                    f"system prompt differs from {version}"))  # fmt: skip
    return issues


def _duplicate_ids(datasets: Mapping[str, Sequence[SFTRecord]]) -> list[Issue]:
    seen: dict[str, str] = {}
    issues = []
    for name, records in datasets.items():
        for record in records:
            if record.id in seen:
                issues.append(Issue(record.id, name, "DUP_ID", f"id also in {seen[record.id]}"))
            seen.setdefault(record.id, name)
    return issues


def _duplicate_texts(datasets: Mapping[str, Sequence[SFTRecord]]) -> list[Issue]:
    """DUP_EXACT across splits (error), DUP_EXACT_IN_SPLIT inside one split (warning)."""
    where: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, records in datasets.items():
        for record in records:
            where[normalized_request(record)].append((name, record.id))
    issues = []
    for places in where.values():
        if len(places) < 2:
            continue
        splits = sorted({name for name, _ in places})
        code = "DUP_EXACT" if len(splits) > 1 else "DUP_EXACT_IN_SPLIT"
        ids = sorted(record_id for _, record_id in places)
        for name, record_id in places:
            issues.append(Issue(record_id, name, code, f"same request text as {ids} in {splits}"))
    return issues


def _smoke_issues(smoke: Sequence[SFTRecord], train: Sequence[SFTRecord]) -> list[Issue]:
    by_id = {record.id: record for record in train}
    return [
        Issue(record.id, "smoke", "SMOKE_NOT_IN_TRAIN", "smoke record differs from train")
        for record in smoke
        if record.split != "train" or by_id.get(record.id) != record
    ]


def _manifest_extras(refs: Mapping[str, ArtifactRef], root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: load_artifact_manifest(root / ref.path, root=root).extra for name, ref in refs.items()
    }


def _holdouts(extras: Mapping[str, Mapping[str, Any]]) -> tuple[set[str], set[str], list[Issue]]:
    """Held-out routes and families declared by the split manifests (they must agree)."""
    routes = {tuple(sorted(extra.get("holdout_routes", []))) for extra in extras.values()}
    families = {tuple(sorted(extra.get("holdout_families", []))) for extra in extras.values()}
    issues = []
    if len(routes) > 1 or len(families) > 1:
        message = "split manifests declare different held-out routes or families"
        issues.append(Issue("<manifests>", None, "HOLDOUT_MISMATCH", message))
    return {r for group in routes for r in group}, {f for group in families for f in group}, issues


def _routes_by_load(ref: ArtifactRef, root: Path) -> dict[str, str]:
    """load_id -> route_id from the load_facts artifact this split was generated from."""
    parents = [a for a in lineage(ref, root=root) if a.kind == "load_facts"]
    if not parents:
        raise QFError(f"{ref.path}: no load_facts artifact among its ancestors")
    facts_ref = read_artifact(root / parents[0].path, "load_facts",
                              supported_versions("load_facts"), root=root)  # fmt: skip
    return {fact.load_id: fact.route_id for fact in load_facts(facts_ref, root)}


def _ood_issues(
    refs: Mapping[str, ArtifactRef], datasets: Mapping[str, Sequence[SFTRecord]], root: Path
) -> list[Issue]:
    """OOD_FAMILY_LEAK / OOD_ROUTE_LEAK: held-out families and routes outside test_ood."""
    extras = _manifest_extras({n: r for n, r in refs.items() if n in SPLIT_FILES}, root)
    routes, families, issues = _holdouts(extras)
    families |= {n for n in TEMPLATE_FAMILIES.names() if TEMPLATE_FAMILIES.get(n)().ood_only}
    route_maps: dict[str, dict[str, str]] = {}
    for name in MAIN_SPLITS:
        for record in datasets.get(name, []):
            if record.template_family in families:
                issues.append(Issue(record.id, name, "OOD_FAMILY_LEAK",
                                    f"held-out family {record.template_family}"))  # fmt: skip
            if not routes:
                continue
            ref = refs[name]
            if ref.sha256 not in route_maps:
                route_maps[ref.sha256] = _routes_by_load(ref, root)
            load_id = record.group_id.removeprefix("load:")
            route = route_maps[ref.sha256].get(load_id)
            if route is None:
                message = f"group {record.group_id} is not a load of the facts"
                issues.append(Issue(record.id, name, "UNKNOWN_LOAD", message))
            elif route in routes:
                issues.append(Issue(record.id, name, "OOD_ROUTE_LEAK", f"held-out route {route}"))
    return issues


def _bench_issues(
    bench: Sequence[SFTRecord], datasets: Mapping[str, Sequence[SFTRecord]]
) -> list[Issue]:
    """BENCH_LEAK: a benchmark id, group or request text also in train or val."""
    seen: dict[str, str] = {}
    for name in ("train", "val"):
        for record in datasets.get(name, []):
            for key in (f"id:{record.id}", f"group:{record.group_id}",
                        f"text:{normalized_request(record)}"):  # fmt: skip
                seen.setdefault(key, name)
    issues = []
    for record in bench:
        for key in (f"id:{record.id}", f"group:{record.group_id}",
                    f"text:{normalized_request(record)}"):  # fmt: skip
            if key in seen:
                message = f"{key.split(':', 1)[0]} also in {seen[key]}"
                issues.append(Issue(record.id, "bench", "BENCH_LEAK", message))
                break
    return issues


def validate_dataset(
    refs: Mapping[str, ArtifactRef], bench: ArtifactRef | None, *, root: Path
) -> ValidationReport:
    """All checks of step 7 over verified artifacts: split files (train, val, test, test_ood),
    optional `smoke` (must repeat train records) and an optional benchmark."""
    issues: list[Issue] = []
    datasets: dict[str, list[SFTRecord]] = {}
    for name, ref in refs.items():
        records, read_issues = read_sft_records(root / ref.path, name)
        datasets[name] = records
        issues += read_issues
    splits: dict[str, list[SFTRecord]] = {n: datasets[n] for n in SPLIT_FILES if n in datasets}
    for name, records in splits.items():
        issues += _record_issues(name, records)
    issues += split_issues(splits) + _duplicate_ids(splits) + _duplicate_texts(splits)
    issues += _ood_issues(refs, splits, root)
    if "smoke" in datasets:
        issues += _smoke_issues(datasets["smoke"], datasets.get("train", []))
    bench_info = None
    if bench is not None:
        bench_records, bench_issues = read_sft_records(root / bench.path, "bench")
        issues += bench_issues + _record_issues("bench", bench_records)
        issues += _bench_issues(bench_records, splits)
        bench_info = {"path": bench.path.as_posix(), "sha256": bench.sha256,
                      "records": len(bench_records)}  # fmt: skip
    return _report(refs, datasets, bench_info, issues)


def _report(
    refs: Mapping[str, ArtifactRef],
    datasets: Mapping[str, Sequence[SFTRecord]],
    bench: dict[str, Any] | None,
    issues: Iterable[Issue],
) -> ValidationReport:
    items = [IssueRecord(record_id=i.record_id, split=i.split, code=i.code, message=i.message)
             for i in issues]  # fmt: skip
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.code] += 1
    return ValidationReport(
        datasets={name: {"path": ref.path.as_posix(), "sha256": ref.sha256,
                         "records": len(datasets.get(name, []))} for name, ref in refs.items()},
        bench=bench,
        counts=dict(sorted(counts.items())),
        errors=[item for item in items if item.code not in _WARNING_CODES],
        warnings=[item for item in items if item.code in _WARNING_CODES],
    )  # fmt: skip


@dataclass(frozen=True)
class ValidationOutcome:
    report: ValidationReport
    ref: ArtifactRef
    run_dir: Path


def run_validation(
    paths: Mapping[str, Path], bench_path: Path | None, *, root: Path, out_path: Path
) -> ValidationOutcome:
    """The stage: verify the artifacts, validate, always write `validation_report.json`
    (a `metrics` artifact whose parents are the checked files) and a run manifest."""
    run = start_run("validate", root)
    refs = {name: read_artifact(path, "sft_dataset", supported_versions("sft_dataset"), root=root)
            for name, path in paths.items()}  # fmt: skip
    bench = None
    if bench_path is not None:
        bench = read_artifact(bench_path, "benchmark", supported_versions("benchmark"), root=root)
    report = validate_dataset(refs, bench, root=root)
    atomic_write_text(out_path, report.model_dump_json(indent=2) + "\n")
    parents = [ref.sha256 for ref in refs.values()] + ([bench.sha256] if bench else [])
    ref = write_artifact(out_path, "metrics", VALIDATION_REPORT_VERSION, run.run_id, parents,
                         root=root)  # fmt: skip
    checked = [*refs.values(), *([bench] if bench else [])]
    run.finish(
        data_hashes={r.path.as_posix(): r.sha256 for r in checked},
        metrics={"passed": report.passed, "counts": report.counts},
    )
    return ValidationOutcome(report=report, ref=ref, run_dir=run.run_dir)
