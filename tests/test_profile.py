from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qf.cli import main
from qf.common import (
    DataValidationError,
    QFError,
    load_artifact_manifest,
    load_yaml_config,
    read_manifest,
    write_artifact,
)
from qf.data import (
    CHECKS,
    EXPECTED_COLUMNS,
    Expectations,
    ProfileReport,
    load_raw_tables,
    profile_raw_dataset,
    profile_tables,
    render_profile_markdown,
)
from tests.conftest import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "raw_mini"
REVISION = "54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07"
RAW_TARGET = Path("data/raw/logistics-operations") / REVISION
CHECK_NAMES = [
    "row_counts", "primary_keys", "foreign_keys", "event_pairs", "route_matches_events",
    "pickup_date_equals_load_date", "transit_days", "value_ranges", "cities", "null_rates",
]  # fmt: skip
REAL_RAW = REPO_ROOT / RAW_TARGET


def fixture_expectations(**overrides: object) -> Expectations:
    data: dict[str, object] = {
        "row_counts": {"loads": 10, "routes": 3, "customers": 3, "delivery_events": 20},
        "weight_lbs_min": 10000,
        "weight_lbs_max": 45000,
        "pieces_max": 28,
        "max_transit_days": 3,
        "load_types": ("Dry Van", "Refrigerated"),
        "freight_types": (
            "General", "Retail", "Consumer Goods", "Food/Beverage", "Automotive", "Electronics",
        ),
        "cities": {"Atlanta": "GA", "Chicago": "IL", "Houston": "TX", "Detroit": "MI",
                   "Los Angeles": "CA"},
    }  # fmt: skip
    data.update(overrides)
    return Expectations.model_validate(data)


Mutation = Callable[[pd.DataFrame], pd.DataFrame]


def mutated_fixture(tmp_path: Path, table: str, mutate: Mutation) -> Path:
    """Copy of the fixture with one table changed; values are kept as the original text."""
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE, raw)
    path = raw / f"{table}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    mutate(frame).to_csv(path, index=False)
    return raw


def profile_of(raw: Path, **overrides: object) -> dict[str, object]:
    report = profile_tables(load_raw_tables(raw), fixture_expectations(**overrides))
    return {check.name: check for check in report.checks}


def set_value(key: str, key_value: str, column: str, value: str) -> Mutation:
    def mutate(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame[key] == key_value, column] = value
        return frame

    return mutate


# --- the clean fixture ---------------------------------------------------------------------


def test_profile_passes_on_clean_fixture() -> None:
    report = profile_tables(load_raw_tables(FIXTURE), fixture_expectations())
    assert [check.name for check in report.checks] == CHECK_NAMES
    assert report.passed, [c for c in report.checks if not c.passed]
    assert report.row_counts == {"loads": 10, "routes": 3, "customers": 3, "delivery_events": 20}


def test_checks_are_data_in_the_documented_order() -> None:
    assert [check.__name__.removeprefix("check_") for check in CHECKS] == CHECK_NAMES


def test_distributions_of_fixture() -> None:
    distributions = profile_tables(load_raw_tables(FIXTURE), fixture_expectations()).distributions
    assert distributions["load_type"] == {"Dry Van": 5, "Refrigerated": 5}
    assert distributions["transit_days"] == {"0": 2, "1": 4, "2": 3, "3": 1}
    assert distributions["weight_lbs_quantiles"]["q0"] == 10000.0
    assert distributions["weight_lbs_quantiles"]["q1"] == 45000.0
    assert distributions["event_type"] == {"Delivery": 10, "Pickup": 10}
    assert list(distributions["load_month"])[0] == "2022-01"
    assert set(distributions["route_id"]) == {"RTE00001", "RTE00002", "RTE00003"}


# --- each check detects its problem --------------------------------------------------------


def test_detects_missing_fk(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "routes", lambda f: f[f["route_id"] != "RTE00003"])
    check = profile_of(raw)["foreign_keys"]
    assert not check.passed
    assert "loads.route_id not in routes" in check.details[0]
    assert "LOAD00000003" in check.details[0]


def test_detects_orphan_event(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", lambda f: f[f["load_id"] != "LOAD00000010"])
    checks = profile_of(raw, row_counts=None)
    assert "delivery_events.load_id not in loads" in checks["foreign_keys"].details[0]
    assert "EVT00000019" in checks["foreign_keys"].details[0]


def test_detects_duplicate_pickup_event(tmp_path: Path) -> None:
    def duplicate_pickup(frame: pd.DataFrame) -> pd.DataFrame:
        extra = frame[frame["event_id"] == "EVT00000001"].assign(event_id="EVT00000099")
        return pd.concat([frame, extra], ignore_index=True)

    raw = mutated_fixture(tmp_path, "delivery_events", duplicate_pickup)
    check = profile_of(raw, row_counts=None)["event_pairs"]
    assert not check.passed
    assert (
        "not exactly one Pickup and one Delivery: 1 record(s), e.g. LOAD00000001" in check.details
    )


def test_detects_duplicate_primary_key(tmp_path: Path) -> None:
    raw = mutated_fixture(
        tmp_path, "customers", set_value("customer_id", "CUST00003", "customer_id", "CUST00002")
    )
    check = profile_of(raw)["primary_keys"]
    assert check.details == ["customers.customer_id duplicated: 1 record(s), e.g. CUST00002"]


def test_detects_unknown_event_type_and_load_without_events(tmp_path: Path) -> None:
    raw = mutated_fixture(
        tmp_path, "delivery_events", set_value("event_id", "EVT00000001", "event_type", "Arrival")
    )
    details = " ".join(profile_of(raw)["event_pairs"].details)
    assert "unknown event_type: 1 record(s), e.g. EVT00000001" in details
    raw_without = mutated_fixture(
        tmp_path / "b", "delivery_events", lambda f: f[f["load_id"] != "LOAD00000004"]
    )
    details = " ".join(profile_of(raw_without, row_counts=None)["event_pairs"].details)
    assert "loads without events: 1 record(s), e.g. LOAD00000004" in details


def test_detects_city_mismatch_between_route_and_event(tmp_path: Path) -> None:
    raw = mutated_fixture(
        tmp_path, "delivery_events", set_value("event_id", "EVT00000004", "location_city", "Denver")
    )
    check = profile_of(raw)["route_matches_events"]
    assert not check.passed
    assert "route destination_city differs from event: 1 record(s), e.g. LOAD00000002" in (
        check.details
    )
    assert check.stats["destination_city_match_rate"] == 0.9


def test_detects_pickup_date_mismatch(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000005", "load_date",
                                                        "2022-06-20"))  # fmt: skip
    check = profile_of(raw)["pickup_date_equals_load_date"]
    assert not check.passed
    assert "LOAD00000005" in check.details[0]
    assert check.stats["mismatches"] == 1


def test_detects_transit_out_of_range(tmp_path: Path) -> None:
    raw = mutated_fixture(
        tmp_path,
        "delivery_events",
        set_value("event_id", "EVT00000006", "scheduled_datetime", "2022-03-20 17:30:00.000000"),
    )
    check = profile_of(raw)["transit_days"]
    assert not check.passed
    assert "outside 0..3 days: 1 record(s), e.g. LOAD00000003" in check.details[0]


def test_detects_out_of_range_weight(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000003", "weight_lbs",
                                                        "50000"))  # fmt: skip
    check = profile_of(raw)["value_ranges"]
    assert not check.passed
    assert check.details == ["weight_lbs out of range: 1 record(s), e.g. LOAD00000003"]


@pytest.mark.parametrize(
    ("table", "key", "key_value", "column", "value", "message"),
    [
        ("loads", "load_id", "LOAD00000001", "pieces", "0", "pieces out of range"),
        ("loads", "load_id", "LOAD00000002", "load_type", "Flatbed", "unexpected load_type"),
        ("customers", "customer_id", "CUST00001", "primary_freight_type", "Toys",
         "unexpected primary_freight_type"),
    ],
)  # fmt: skip
def test_detects_other_range_violations(
    tmp_path: Path, table: str, key: str, key_value: str, column: str, value: str, message: str
) -> None:
    raw = mutated_fixture(tmp_path, table, set_value(key, key_value, column, value))
    details = profile_of(raw)["value_ranges"].details
    assert details == [f"{message}: 1 record(s), e.g. {key_value}"]


def test_detects_city_problems(tmp_path: Path) -> None:
    raw = mutated_fixture(
        tmp_path, "routes", set_value("route_id", "RTE00003", "destination_state", "NV")
    )
    details = " ".join(profile_of(raw)["cities"].details)
    assert "unexpected state for city: Los Angeles: NV (expected CA)" in details
    raw_two_states = mutated_fixture(
        tmp_path / "b", "routes", set_value("route_id", "RTE00001", "destination_state", "IN")
    )
    details = " ".join(profile_of(raw_two_states)["cities"].details)
    assert "city with more than one state: 1 record(s), e.g. Chicago" in details
    checks = profile_of(FIXTURE, cities={"Atlanta": "GA", "Miami": "FL"})
    details = " ".join(checks["cities"].details)
    assert "expected cities absent from routes: Miami" in details
    assert "unexpected cities in routes: Chicago, Detroit, Houston, Los Angeles" in details


def test_nulls_are_reported_without_crashing_other_checks(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000006", "weight_lbs", ""))
    checks = profile_of(raw)
    assert checks["null_rates"].details == ["loads.weight_lbs: null rate 10.0000% > 0.0000%"]
    assert checks["value_ranges"].passed  # a missing weight is not a range violation


def test_row_count_mismatch_and_unconfigured(tmp_path: Path) -> None:
    checks = profile_of(FIXTURE, row_counts={"loads": 11})
    assert checks["row_counts"].details == ["loads: 10 rows, expected 11"]
    assert profile_of(FIXTURE, row_counts=None)["row_counts"].details == ["not configured"]


# --- loader --------------------------------------------------------------------------------


def test_wrong_columns_raise(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", lambda f: f.drop(columns=["pieces"]).assign(extra="x"))
    with pytest.raises(DataValidationError) as info:
        load_raw_tables(raw)
    issue = info.value.issues[0]
    assert (issue.record_id, issue.code) == ("loads.csv", "COLUMNS")
    assert "missing ['pieces'], unexpected ['extra']" in issue.message


def test_column_order_does_not_matter(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "routes", lambda f: f[list(reversed(f.columns))])
    assert list(load_raw_tables(raw).routes.columns) == list(reversed(EXPECTED_COLUMNS["routes"]))


def test_bad_date_names_the_record(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000007", "load_date",
                                                        "14/02/2023"))  # fmt: skip
    with pytest.raises(DataValidationError) as info:
        load_raw_tables(raw)
    issue = info.value.issues[0]
    assert (issue.record_id, issue.code) == ("LOAD00000007", "DATE_FORMAT")
    assert "load_date='14/02/2023'" in issue.message


def test_non_numeric_value_is_a_validation_error(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000001", "pieces", "ten"))
    with pytest.raises(DataValidationError) as info:
        load_raw_tables(raw)
    assert info.value.issues[0].code == "DTYPE"


def test_missing_table_file(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / "raw")
    (tmp_path / "raw" / "customers.csv").unlink()
    with pytest.raises(QFError, match="customers.csv is missing"):
        load_raw_tables(tmp_path / "raw")


def test_expectations_validation() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        fixture_expectations(weight_lbs_min=50000)
    with pytest.raises(ValueError, match="may only name tables"):
        fixture_expectations(row_counts={"drivers": 150})


def test_repository_expectations_config() -> None:
    expectations = load_yaml_config(REPO_ROOT / "configs/data/expectations.yaml", Expectations)
    assert len(expectations.cities) == 20
    assert expectations.row_counts == {
        "loads": 85410, "routes": 58, "customers": 200, "delivery_events": 170820,
    }  # fmt: skip
    assert expectations.max_transit_days == 3


# --- report and use case -------------------------------------------------------------------


def test_markdown_report_lists_checks_and_failures(tmp_path: Path) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000003", "weight_lbs",
                                                        "50000"))  # fmt: skip
    report = profile_tables(load_raw_tables(raw), fixture_expectations())
    markdown = render_profile_markdown(report)
    assert "Verdict: **FAILED: value_ranges**" in markdown
    assert "| value_ranges | FAIL | weight_lbs out of range" in markdown
    assert "| cities | PASS |" in markdown
    assert "### transit_days" in markdown


def _install_raw(project: Path, source: Path = FIXTURE) -> None:
    shutil.copytree(source, project / RAW_TARGET)
    write_artifact(
        project / RAW_TARGET, "raw_dataset", "logistics_ops_csv_v1", "run-f", root=project
    )


def test_profile_raw_dataset_writes_report_artifact_and_manifest(fake_project: Path) -> None:
    _install_raw(fake_project)
    raw_ref = load_artifact_manifest(RAW_TARGET).ref
    outcome = profile_raw_dataset(
        RAW_TARGET, fixture_expectations(), root=fake_project,
        out_dir=fake_project / "data/processed", expectations_config={"pieces_max": 28},
    )  # fmt: skip
    assert outcome.report.passed
    stored = ProfileReport.model_validate_json(outcome.json_path.read_text(encoding="utf-8"))
    assert stored.raw_dataset == {"path": RAW_TARGET.as_posix(), "sha256": raw_ref.sha256}
    assert outcome.markdown_path.read_text(encoding="utf-8").startswith("# Raw data profile")
    assert (outcome.artifact.kind, outcome.artifact.schema_version) == (
        "metrics",
        "profile_report_v1",
    )
    assert outcome.artifact.parents == (raw_ref.sha256,)
    manifest = read_manifest(outcome.run_dir)
    assert (manifest.kind, manifest.status) == ("profile", "completed")
    assert manifest.metrics == {"all_checks_passed": True, "failed_checks": []}
    assert manifest.data_hashes == {RAW_TARGET.as_posix(): raw_ref.sha256}


def test_profile_refuses_raw_data_changed_after_fetch(fake_project: Path) -> None:
    _install_raw(fake_project)
    (fake_project / RAW_TARGET / "routes.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        profile_raw_dataset(
            RAW_TARGET, fixture_expectations(), root=fake_project,
            out_dir=fake_project / "data/processed", expectations_config={},
        )  # fmt: skip


# --- CLI -----------------------------------------------------------------------------------


@pytest.fixture
def profile_project(fake_project: Path) -> Path:
    configs = fake_project / "configs" / "data"
    configs.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs/data/source.yaml", configs / "source.yaml")
    expectations = fixture_expectations().model_dump(mode="json")
    (configs / "expectations.yaml").write_text(yaml.safe_dump(expectations), encoding="utf-8")
    return fake_project


def test_cli_profile_passes(profile_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install_raw(profile_project)
    assert main(["data", "profile"]) == 0
    out = capsys.readouterr().out
    assert "PASS  route_matches_events" in out
    report = json.loads((profile_project / "data/processed/profile_report.json").read_text())
    assert [check["name"] for check in report["checks"]] == CHECK_NAMES


def test_cli_profile_fails_with_exit_1_and_still_writes_report(
    profile_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = mutated_fixture(tmp_path, "loads", set_value("load_id", "LOAD00000003", "weight_lbs",
                                                        "50000"))  # fmt: skip
    _install_raw(profile_project, raw)
    assert main(["data", "profile"]) == 1
    captured = capsys.readouterr()
    assert "FAIL  value_ranges" in captured.out
    assert "weight_lbs out of range: 1 record(s), e.g. LOAD00000003" in captured.out
    assert "profile failed: value_ranges" in captured.err
    assert (profile_project / "data/processed/profile_report.md").is_file()


def test_cli_profile_before_fetch(
    profile_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "profile"]) == 1
    assert "missing or unreadable artifact manifest" in capsys.readouterr().err


# --- real data -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.is_dir(), reason="raw dataset not fetched: run `qf data fetch`")
def test_profile_real_data() -> None:
    expectations = load_yaml_config(REPO_ROOT / "configs/data/expectations.yaml", Expectations)
    report = profile_tables(load_raw_tables(REAL_RAW), expectations)
    assert report.passed, [(c.name, c.details) for c in report.checks if not c.passed]
    assert report.distributions["transit_days"] == {"0": 16835, "1": 44386, "2": 23546, "3": 643}
