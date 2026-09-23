"""Stage `profile`: integrity checks and distributions of the raw tables (plan step 3).

Each check is a separate `check_*` function returning a `CheckResult`; `CHECKS` (data, not
code paths) lists them in order. The report is written as JSON (an artifact of kind `metrics`,
parent: the raw dataset) plus a human-readable Markdown file. Any failed check makes the stage
fail, so the pipeline stops before generating data from unexpected input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qf.common import (
    ArtifactRef,
    StrictConfig,
    atomic_write_text,
    read_artifact,
    start_run,
    write_artifact,
)
from qf.contracts import supported_versions
from qf.data.raw_tables import DTYPES, PRIMARY_KEYS, RawTables, load_raw_tables

__all__ = [
    "CHECKS",
    "PROFILE_REPORT_VERSION",
    "USED_COLUMNS",
    "CheckResult",
    "Expectations",
    "ProfileOutcome",
    "ProfileReport",
    "compute_distributions",
    "profile_raw_dataset",
    "profile_tables",
    "render_profile_markdown",
]

PROFILE_REPORT_VERSION = "profile_report_v1"
PROFILE_JSON = "profile_report.json"
PROFILE_MARKDOWN = "profile_report.md"
PICKUP, DELIVERY = "Pickup", "Delivery"
_MAX_EXAMPLES = 10
_WEIGHT_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
_RUN_KIND = "profile"

# Columns the later stages use; their null rate must not exceed `Expectations.max_null_rate`.
USED_COLUMNS: dict[str, tuple[str, ...]] = {
    "loads": (
        "load_id",
        "customer_id",
        "route_id",
        "load_date",
        "load_type",
        "weight_lbs",
        "pieces",
    ),
    "routes": ("route_id", "origin_city", "origin_state", "destination_city", "destination_state"),
    "customers": ("customer_id", "customer_name", "primary_freight_type"),
    "delivery_events": (
        "event_id",
        "load_id",
        "event_type",
        "scheduled_datetime",
        "location_city",
        "location_state",
    ),
}


class Expectations(StrictConfig):
    """What the raw tables must look like (configs/data/expectations.yaml)."""

    row_counts: dict[str, int] | None = None
    weight_lbs_min: int = Field(gt=0)
    weight_lbs_max: int = Field(gt=0)
    pieces_max: int = Field(ge=1)
    max_transit_days: int = Field(ge=0)
    load_types: tuple[str, ...] = Field(min_length=1)
    freight_types: tuple[str, ...] = Field(min_length=1)
    cities: dict[str, str] = Field(min_length=1)  # city -> two-letter state code
    max_null_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> Expectations:
        if self.weight_lbs_min > self.weight_lbs_max:
            raise ValueError("weight_lbs_min must not exceed weight_lbs_max")
        if self.row_counts is not None and set(self.row_counts) - set(DTYPES):
            raise ValueError(f"row_counts may only name tables {sorted(DTYPES)}")
        return self


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    details: list[str] = Field(default_factory=list)  # what is wrong, with example ids
    stats: dict[str, Any] = Field(default_factory=dict)


class ProfileReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: Literal["profile_report_v1"] = "profile_report_v1"
    generated_at: datetime
    raw_dataset: dict[str, str] = Field(default_factory=dict)  # path and sha256 of the input
    row_counts: dict[str, int]
    checks: list[CheckResult]
    distributions: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]


# --- helpers -------------------------------------------------------------------------------


def _examples(ids: Iterable[Any]) -> str:
    unique = sorted({str(value) for value in ids})
    shown = ", ".join(unique[:_MAX_EXAMPLES])
    return shown + (
        f" (+{len(unique) - _MAX_EXAMPLES} more)" if len(unique) > _MAX_EXAMPLES else ""
    )


def _problem(what: str, ids: pd.Series) -> str:
    return f"{what}: {ids.nunique()} record(s), e.g. {_examples(ids)}"


def _counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _events_by_load(t: RawTables) -> pd.DataFrame:
    """One row per load that has exactly one Pickup and one Delivery event."""
    events = t.delivery_events[t.delivery_events["event_type"].isin([PICKUP, DELIVERY])]
    events = events.drop_duplicates(subset=["load_id", "event_type"], keep=False)
    columns = ["scheduled_datetime", "location_city", "location_state"]
    wide = events.pivot(index="load_id", columns="event_type", values=columns)
    pairs = cast(list[tuple[str, str]], list(wide.columns))
    wide.columns = [f"{event.lower()}_{field}" for field, event in pairs]
    wide = wide.dropna(subset=["pickup_location_city", "delivery_location_city"], how="any")
    return wide.reset_index()


def _loads_with_events(t: RawTables) -> pd.DataFrame:
    return t.loads.merge(_events_by_load(t), on="load_id", how="inner")


def _transit_days(merged: pd.DataFrame) -> pd.Series:
    pickup = pd.to_datetime(merged["pickup_scheduled_datetime"]).dt.normalize()
    delivery = pd.to_datetime(merged["delivery_scheduled_datetime"]).dt.normalize()
    return (delivery - pickup).dt.days


# --- checks --------------------------------------------------------------------------------


def check_row_counts(t: RawTables, exp: Expectations) -> CheckResult:
    actual = t.row_counts()
    if exp.row_counts is None:
        return CheckResult(name="row_counts", passed=True, details=["not configured"], stats=actual)
    details = [
        f"{table}: {actual[table]} rows, expected {expected}"
        for table, expected in exp.row_counts.items()
        if actual[table] != expected
    ]
    return CheckResult(name="row_counts", passed=not details, details=details, stats=actual)


def check_primary_keys(t: RawTables, exp: Expectations) -> CheckResult:
    details = []
    for table, key in PRIMARY_KEYS.items():
        ids = getattr(t, table)[key]
        duplicated = ids[ids.duplicated(keep=False) & ids.notna()]
        if not duplicated.empty:
            details.append(_problem(f"{table}.{key} duplicated", duplicated))
    return CheckResult(name="primary_keys", passed=not details, details=details)


def check_foreign_keys(t: RawTables, exp: Expectations) -> CheckResult:
    loads, events = t.loads, t.delivery_events
    relations = (
        ("loads.route_id not in routes", loads, "route_id", t.routes["route_id"], "load_id"),
        ("loads.customer_id not in customers", loads, "customer_id", t.customers["customer_id"],
         "load_id"),
        ("delivery_events.load_id not in loads", events, "load_id", loads["load_id"], "event_id"),
    )  # fmt: skip
    details = []
    for what, frame, column, targets, id_column in relations:
        orphans = frame.loc[~frame[column].isin(targets), id_column]
        if not orphans.empty:
            details.append(_problem(what, orphans))
    return CheckResult(name="foreign_keys", passed=not details, details=details)


def check_event_pairs(t: RawTables, exp: Expectations) -> CheckResult:
    events = t.delivery_events
    details = []
    unknown = events.loc[~events["event_type"].isin([PICKUP, DELIVERY]), "event_id"]
    if not unknown.empty:
        details.append(_problem("unknown event_type", unknown))
    counts = (
        events.groupby(["load_id", "event_type"]).size().unstack(fill_value=0)
        .reindex(columns=[PICKUP, DELIVERY], fill_value=0)
    )  # fmt: skip
    wrong = counts[(counts[PICKUP] != 1) | (counts[DELIVERY] != 1)]
    if not wrong.empty:
        details.append(_problem("not exactly one Pickup and one Delivery", wrong.index.to_series()))
    without = t.loads.loc[~t.loads["load_id"].isin(events["load_id"]), "load_id"]
    if not without.empty:
        details.append(_problem("loads without events", without))
    return CheckResult(name="event_pairs", passed=not details, details=details)


def check_route_matches_events(t: RawTables, exp: Expectations) -> CheckResult:
    merged = _loads_with_events(t).merge(t.routes, on="route_id", how="inner")
    comparisons = {
        "origin_city": ("origin_city", "pickup_location_city"),
        "origin_state": ("origin_state", "pickup_location_state"),
        "destination_city": ("destination_city", "delivery_location_city"),
        "destination_state": ("destination_state", "delivery_location_state"),
    }
    details: list[str] = []
    stats: dict[str, Any] = {"compared_loads": len(merged)}
    for label, (route_column, event_column) in comparisons.items():
        mismatch = merged[route_column] != merged[event_column]
        stats[f"{label}_match_rate"] = round(1 - float(mismatch.mean()), 6) if len(merged) else 1.0
        if mismatch.any():
            details.append(
                _problem(f"route {label} differs from event", merged.loc[mismatch, "load_id"])
            )
    return CheckResult(
        name="route_matches_events", passed=not details, details=details, stats=stats
    )


def check_pickup_date_equals_load_date(t: RawTables, exp: Expectations) -> CheckResult:
    merged = _loads_with_events(t)
    pickup_date = pd.to_datetime(merged["pickup_scheduled_datetime"]).dt.normalize()
    mismatch = (pickup_date != merged["load_date"]).fillna(True)
    details = []
    if mismatch.any():
        details.append(
            _problem("pickup date differs from load_date", merged.loc[mismatch, "load_id"])
        )
    return CheckResult(
        name="pickup_date_equals_load_date",
        passed=not details,
        details=details,
        stats={"compared_loads": len(merged), "mismatches": int(mismatch.sum())},
    )


def check_transit_days(t: RawTables, exp: Expectations) -> CheckResult:
    merged = _loads_with_events(t)
    days = _transit_days(merged)
    bad = (days < 0) | (days > exp.max_transit_days)
    details = []
    if bad.any():
        what = f"delivery - pickup outside 0..{exp.max_transit_days} days"
        details.append(_problem(what, merged.loc[bad, "load_id"]))
    return CheckResult(
        name="transit_days", passed=not details, details=details, stats={"days": _counts(days)}
    )


def check_value_ranges(t: RawTables, exp: Expectations) -> CheckResult:
    loads, customers = t.loads, t.customers
    weight_ok = loads["weight_lbs"].between(exp.weight_lbs_min, exp.weight_lbs_max)
    freight_ok = customers["primary_freight_type"].isin(exp.freight_types)
    rules: tuple[tuple[str, pd.DataFrame, str, pd.Series, str], ...] = (
        ("weight_lbs out of range", loads, "weight_lbs", weight_ok, "load_id"),
        ("pieces out of range", loads, "pieces", loads["pieces"].between(1, exp.pieces_max),
         "load_id"),
        ("unexpected load_type", loads, "load_type", loads["load_type"].isin(exp.load_types),
         "load_id"),
        ("unexpected primary_freight_type", customers, "primary_freight_type", freight_ok,
         "customer_id"),
    )  # fmt: skip
    details = []
    for what, frame, column, ok, id_column in rules:
        # Missing values are the null-rate check's business, not a range violation.
        bad = ~ok.fillna(True).astype(bool) & frame[column].notna()
        if bad.any():
            details.append(_problem(what, frame.loc[bad, id_column]))
    return CheckResult(name="value_ranges", passed=not details, details=details)


def check_cities(t: RawTables, exp: Expectations) -> CheckResult:
    routes = t.routes
    pairs = pd.concat(
        [
            routes[["origin_city", "origin_state"]].set_axis(["city", "state"], axis=1),
            routes[["destination_city", "destination_state"]].set_axis(["city", "state"], axis=1),
        ]
    ).drop_duplicates()
    details = []
    states_per_city = pairs.groupby("city")["state"].nunique()
    ambiguous = states_per_city[states_per_city > 1].index.to_series()
    if not ambiguous.empty:
        details.append(_problem("city with more than one state", ambiguous))
    cities = set(pairs["city"].dropna())
    if missing := sorted(set(exp.cities) - cities):
        details.append(f"expected cities absent from routes: {', '.join(missing)}")
    if unexpected := sorted(cities - set(exp.cities)):
        details.append(f"unexpected cities in routes: {', '.join(unexpected)}")
    wrong_state = [
        f"{city}: {state} (expected {exp.cities[city]})"
        for city, state in zip(pairs["city"], pairs["state"], strict=True)
        if city in exp.cities and state != exp.cities[city]
    ]
    if wrong_state:
        details.append(f"unexpected state for city: {'; '.join(sorted(wrong_state))}")
    return CheckResult(
        name="cities", passed=not details, details=details, stats={"cities": len(cities)}
    )


def check_null_rates(t: RawTables, exp: Expectations) -> CheckResult:
    details: list[str] = []
    stats: dict[str, Any] = {}
    for table, columns in USED_COLUMNS.items():
        frame = getattr(t, table)
        for column in columns:
            rate = float(frame[column].isna().mean()) if len(frame) else 0.0
            stats[f"{table}.{column}"] = round(rate, 6)
            if rate > exp.max_null_rate:
                details.append(f"{table}.{column}: null rate {rate:.4%} > {exp.max_null_rate:.4%}")
    return CheckResult(name="null_rates", passed=not details, details=details, stats=stats)


Check = Callable[[RawTables, Expectations], CheckResult]
CHECKS: tuple[Check, ...] = (
    check_row_counts,
    check_primary_keys,
    check_foreign_keys,
    check_event_pairs,
    check_route_matches_events,
    check_pickup_date_equals_load_date,
    check_transit_days,
    check_value_ranges,
    check_cities,
    check_null_rates,
)


# --- report --------------------------------------------------------------------------------


def compute_distributions(t: RawTables) -> dict[str, Any]:
    loads = t.loads
    weights = loads["weight_lbs"].dropna().astype("float64")
    quantiles = weights.quantile(list(_WEIGHT_QUANTILES)) if len(weights) else pd.Series()
    return {
        "load_type": _counts(loads["load_type"]),
        "load_month": _counts(loads["load_date"].dt.strftime("%Y-%m")),
        "route_id": _counts(loads["route_id"]),
        "pieces": _counts(loads["pieces"]),
        "weight_lbs_quantiles": {f"q{q:g}": float(v) for q, v in quantiles.items()},
        "primary_freight_type": _counts(t.customers["primary_freight_type"]),
        "event_type": _counts(t.delivery_events["event_type"]),
        "transit_days": _counts(_transit_days(_loads_with_events(t))),
    }


def profile_tables(t: RawTables, exp: Expectations) -> ProfileReport:
    return ProfileReport(
        generated_at=datetime.now(UTC),
        row_counts=t.row_counts(),
        checks=[check(t, exp) for check in CHECKS],
        distributions=compute_distributions(t),
    )


def render_profile_markdown(report: ProfileReport) -> str:
    verdict = "all checks passed" if report.passed else f"FAILED: {', '.join(report.failed_checks)}"
    lines = [
        "# Raw data profile",
        "",
        f"- Generated: {report.generated_at.isoformat()}",
        f"- Raw dataset: `{report.raw_dataset.get('path', '?')}` "
        f"(sha256 `{report.raw_dataset.get('sha256', '?')[:12]}`)",
        f"- Verdict: **{verdict}**",
        f"- Rows: {', '.join(f'{k} {v:,}' for k, v in report.row_counts.items())}",
        "",
        "## Checks",
        "",
        "| Check | Result | Details |",
        "|---|---|---|",
    ]
    for check in report.checks:
        details = "<br>".join(check.details) if check.details else ""
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {details} |")
    lines += ["", "## Distributions", ""]
    for name, values in report.distributions.items():
        lines += [f"### {name}", "", "| Value | Count |", "|---|---|"]
        lines += [f"| {key} | {value:,} |" for key, value in values.items()]
        lines.append("")
    return "\n".join(lines)


# --- use case ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileOutcome:
    report: ProfileReport
    json_path: Path
    markdown_path: Path
    artifact: ArtifactRef
    run_dir: Path


def profile_raw_dataset(
    raw_path: Path,
    expectations: Expectations,
    *,
    root: Path,
    out_dir: Path,
    expectations_config: Mapping[str, Any],
) -> ProfileOutcome:
    """Profile the verified raw dataset at `raw_path`; always write the report, even on failure."""
    run = start_run(_RUN_KIND, root)
    raw_ref = read_artifact(raw_path, "raw_dataset", supported_versions("raw_dataset"), root=root)
    report = profile_tables(load_raw_tables(root / raw_ref.path), expectations).model_copy(
        update={"raw_dataset": {"path": raw_ref.path.as_posix(), "sha256": raw_ref.sha256}}
    )
    json_path, markdown_path = out_dir / PROFILE_JSON, out_dir / PROFILE_MARKDOWN
    atomic_write_text(json_path, report.model_dump_json(indent=2) + "\n")
    atomic_write_text(markdown_path, render_profile_markdown(report))
    artifact = write_artifact(
        json_path, "metrics", PROFILE_REPORT_VERSION, run.run_id, parents=[raw_ref.sha256],
        root=root,
    )  # fmt: skip
    run.finish(
        packages=["pandas"],
        data_hashes={raw_ref.path.as_posix(): raw_ref.sha256},
        config=dict(expectations_config),
        metrics={"all_checks_passed": report.passed, "failed_checks": report.failed_checks},
    )
    return ProfileOutcome(report, json_path, markdown_path, artifact, run.run_dir)
