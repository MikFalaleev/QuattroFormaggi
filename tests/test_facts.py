from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from qf.cli import main
from qf.common import (
    ArtifactRef,
    DataValidationError,
    QFError,
    load_artifact_manifest,
    read_artifact,
    read_manifest,
    sha256_file,
    write_artifact,
)
from qf.contracts import FactsBuilder, LoadFacts, supported_versions
from qf.data import (
    FACTS_BUILDERS,
    LogisticsOpsFactsBuilder,
    build_facts_artifact,
    build_load_facts,
    load_city_map,
    load_facts,
    load_raw_tables,
)
from qf.domain import CITIES
from tests.conftest import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "raw_mini"
CITY_MAP = REPO_ROOT / "configs" / "data" / "city_map_ru_v1.yaml"
RAW_TARGET = Path("data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07")
OUT = Path("data/processed/load_facts.jsonl")

Mutation = Callable[[pd.DataFrame], pd.DataFrame]


def fixture_facts(raw: Path = FIXTURE) -> list[LoadFacts]:
    return build_load_facts(load_raw_tables(raw), load_city_map(CITY_MAP))


def mutated_fixture(tmp_path: Path, table: str, mutate: Mutation) -> Path:
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE, raw)
    path = raw / f"{table}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    mutate(frame).to_csv(path, index=False)
    return raw


def codes_of(raw: Path) -> list[tuple[str, str]]:
    with pytest.raises(DataValidationError) as caught:
        fixture_facts(raw)
    return [(issue.record_id, issue.code) for issue in caught.value.issues]


# --- building ------------------------------------------------------------------------------


def test_facts_from_fixture() -> None:
    facts = fixture_facts()
    assert len(facts) == len(pd.read_csv(FIXTURE / "loads.csv")) == 10
    first = facts[0]
    assert first.model_dump(mode="json") == {
        "load_id": "LOAD00000001", "route_id": "RTE00001", "customer_id": "CUST00002",
        "shipper_name": "National Retail", "cargo_category": "retail",
        "equipment_type": "dry_van", "pieces": 1, "weight_lbs": 10000,
        "origin": {"city": "Воронеж", "region": "Воронежская область"},  # Atlanta
        "destination": {"city": "Казань", "region": "Республика Татарстан"},  # Chicago
        "pickup_date": "2022-01-01", "delivery_date": "2022-01-02", "load_month": "2022-01",
    }  # fmt: skip
    for fact in facts:
        assert fact.origin.city in CITIES and fact.destination.city in CITIES
        assert all(value is not None for value in fact.model_dump().values())


def test_facts_have_no_financial_fields() -> None:
    forbidden = re.compile(r"revenue|surcharge|rate|distance|charge|mile|price|cost")
    assert [name for name in LoadFacts.model_fields if forbidden.search(name)] == []


def test_places_are_shared_objects() -> None:
    facts = fixture_facts()
    by_city = {fact.origin.city: fact.origin for fact in facts}
    assert all(fact.origin is by_city[fact.origin.city] for fact in facts)


def test_facts_sorted_and_deterministic(tmp_path: Path) -> None:
    def shuffle(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sample(frac=1.0, random_state=3)

    shuffled = mutated_fixture(tmp_path, "loads", shuffle)
    first, second = fixture_facts(shuffled), fixture_facts(FIXTURE)
    assert [fact.load_id for fact in first] == sorted(fact.load_id for fact in first)
    as_jsonl = ["".join(f.model_dump_json() + "\n" for f in facts) for facts in (first, second)]
    assert as_jsonl[0] == as_jsonl[1]


def test_mapping_unknown_value_raises(tmp_path: Path) -> None:
    def flatbed(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["load_id"] == "LOAD00000003", "load_type"] = "Flatbed"
        return frame

    def chemicals(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["customer_id"] == "CUST00003", "primary_freight_type"] = "Chemicals"
        return frame

    assert codes_of(mutated_fixture(tmp_path / "a", "loads", flatbed)) == [
        ("LOAD00000003", "UNKNOWN_VALUE")
    ]
    bad = codes_of(mutated_fixture(tmp_path / "b", "customers", chemicals))
    assert bad and {code for _, code in bad} == {"UNKNOWN_VALUE"}


def test_every_broken_load_is_reported(tmp_path: Path) -> None:
    def break_loads(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["load_id"] == "LOAD00000001", "route_id"] = "RTE09999"
        frame.loc[frame["load_id"] == "LOAD00000002", "customer_id"] = "CUST09999"
        frame.loc[frame["load_id"] == "LOAD00000004", "pieces"] = ""
        return frame

    raw = mutated_fixture(tmp_path, "loads", break_loads)
    events = pd.read_csv(raw / "delivery_events.csv", dtype=str, keep_default_na=False)
    lone_pickup = events[
        ~((events["load_id"] == "LOAD00000005") & (events["event_type"] == "Delivery"))
    ]
    lone_pickup.to_csv(raw / "delivery_events.csv", index=False)
    assert codes_of(raw) == [
        ("LOAD00000001", "MISSING_ROUTE"),
        ("LOAD00000002", "MISSING_CUSTOMER"),
        ("LOAD00000004", "MISSING_VALUE"),
        ("LOAD00000005", "EVENTS"),
    ]


def test_value_rejected_by_contract_is_reported(tmp_path: Path) -> None:
    def zero_weight(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["load_id"] == "LOAD00000006", "weight_lbs"] = "0"
        return frame

    raw = mutated_fixture(tmp_path, "loads", zero_weight)
    assert codes_of(raw) == [("LOAD00000006", "INVALID")]


def test_city_outside_the_map_is_reported(tmp_path: Path) -> None:
    def boston(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[frame["route_id"] == "RTE00001", "origin_city"] = "Boston"
        return frame

    bad = codes_of(mutated_fixture(tmp_path, "routes", boston))
    assert bad and {code for _, code in bad} == {"UNKNOWN_CITY"}


def test_duplicate_keys_are_explicit_error(tmp_path: Path) -> None:
    def duplicate(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([frame, frame.head(1)])

    with pytest.raises(QFError, match="duplicate keys"):
        fixture_facts(mutated_fixture(tmp_path, "routes", duplicate))


def test_contract_rejects_delivery_before_pickup() -> None:
    fact = fixture_facts()[0]
    with pytest.raises(ValidationError, match="before pickup"):
        LoadFacts.model_validate(
            {**fact.model_dump(), "delivery_date": fact.pickup_date.replace(year=2021)}
        )
    same_city = fact.model_copy(update={"destination": fact.origin})  # allowed: city delivery
    assert same_city.origin == same_city.destination


# --- stage, artifact, CLI ------------------------------------------------------------------


def _install_project(project: Path) -> None:
    shutil.copytree(FIXTURE, project / RAW_TARGET)
    write_artifact(
        project / RAW_TARGET, "raw_dataset", "logistics_ops_csv_v1", "run-f", root=project
    )
    configs = project / "configs" / "data"
    configs.mkdir(parents=True)
    for name in ("source.yaml", "facts.yaml", "city_map_ru_v1.yaml"):
        shutil.copy(REPO_ROOT / "configs" / "data" / name, configs / name)


def _builder() -> FactsBuilder:
    config = LogisticsOpsFactsBuilder.Config(city_map=Path("configs/data/city_map_ru_v1.yaml"))
    return LogisticsOpsFactsBuilder(config)


def test_builder_is_registered_and_satisfies_port() -> None:
    assert FACTS_BUILDERS.get("logistics_operations_v1") is LogisticsOpsFactsBuilder
    assert isinstance(_builder(), FactsBuilder)
    with pytest.raises(ValidationError, match="relative"):
        LogisticsOpsFactsBuilder.Config(city_map=Path("/etc/city_map.yaml"))


def test_stage_writes_artifact_and_run_manifest(fake_project: Path) -> None:
    _install_project(fake_project)
    raw_ref = load_artifact_manifest(RAW_TARGET).ref
    result = build_facts_artifact(
        _builder(), RAW_TARGET, root=fake_project, out_path=fake_project / OUT,
        config={"builder": {"name": "logistics_operations_v1"}},
    )  # fmt: skip
    ref = read_artifact(OUT, "load_facts", supported_versions("load_facts"))
    assert ref == result.ref
    assert (ref.kind, ref.schema_version, ref.parents) == (
        "load_facts", "load_facts_v1", (raw_ref.sha256,),
    )  # fmt: skip
    assert load_facts(ref, fake_project) == fixture_facts()
    assert len((fake_project / OUT).read_text(encoding="utf-8").splitlines()) == result.count == 10
    city_map_sha = sha256_file(fake_project / "configs/data/city_map_ru_v1.yaml")
    manifest = read_manifest(result.run_dir)
    assert (manifest.kind, manifest.status, manifest.metrics) == (
        "facts", "completed", {"load_facts": 10},
    )  # fmt: skip
    assert manifest.data_hashes == {
        RAW_TARGET.as_posix(): raw_ref.sha256,
        "configs/data/city_map_ru_v1.yaml": city_map_sha,
        OUT.as_posix(): ref.sha256,
    }
    assert load_artifact_manifest(OUT).extra == {
        "count": 10, "inputs": {"configs/data/city_map_ru_v1.yaml": city_map_sha},
    }  # fmt: skip


def test_builder_refuses_unknown_raw_format(fake_project: Path) -> None:
    _install_project(fake_project)
    ref = load_artifact_manifest(RAW_TARGET).ref.model_copy(update={"schema_version": "other_v9"})
    with pytest.raises(QFError, match="cannot read raw format 'other_v9'"):
        _builder().build(ref, fake_project)


class _DuplicatingBuilder:
    def build(self, raw: ArtifactRef, root: Path) -> list[LoadFacts]:
        fact = fixture_facts()[0]
        return [fact, fact]

    def inputs(self, root: Path) -> dict[str, str]:
        return {}


def test_stage_refuses_duplicate_load_ids(fake_project: Path) -> None:
    _install_project(fake_project)
    with pytest.raises(QFError, match="duplicate load_id in facts: LOAD00000001"):
        build_facts_artifact(
            _DuplicatingBuilder(), RAW_TARGET, root=fake_project, out_path=fake_project / OUT,
            config={},
        )  # fmt: skip
    assert not (fake_project / OUT).exists()


def test_stage_refuses_changed_raw_data(fake_project: Path) -> None:
    _install_project(fake_project)
    (fake_project / RAW_TARGET / "loads.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        build_facts_artifact(
            _builder(), RAW_TARGET, root=fake_project, out_path=fake_project / OUT, config={}
        )


def test_load_facts_reports_bad_line(fake_project: Path) -> None:
    _install_project(fake_project)
    result = build_facts_artifact(
        _builder(), RAW_TARGET, root=fake_project, out_path=fake_project / OUT, config={}
    )
    path = fake_project / OUT
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[1].replace('"pieces":13', '"pieces":"13"')]) + "\n")
    with pytest.raises(DataValidationError, match=r"load_facts.jsonl:2: SCHEMA"):
        load_facts(result.ref, fake_project)


def test_cli_facts(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install_project(fake_project)
    assert main(["data", "facts"]) == 0
    out = capsys.readouterr().out
    assert "Built 10 load facts: data/processed/load_facts.jsonl" in out
    manifest = load_artifact_manifest(OUT)
    run_dir = fake_project / "runs" / manifest.ref.producer_run_id
    assert read_manifest(run_dir).config == {
        "builder": {"name": "logistics_operations_v1",
                    "city_map": "configs/data/city_map_ru_v1.yaml"},
    }  # fmt: skip


def test_cli_facts_before_fetch(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install_project(fake_project)
    shutil.rmtree(fake_project / RAW_TARGET)
    assert main(["data", "facts"]) == 1
    assert "qf:" in capsys.readouterr().err


# --- real data -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not (REPO_ROOT / RAW_TARGET).exists(), reason="raw dataset not fetched")
def test_facts_real_count() -> None:
    facts = fixture_facts(REPO_ROOT / RAW_TARGET)
    assert len(facts) == 85_410
    assert len({fact.load_id for fact in facts}) == 85_410
    assert {fact.origin.city for fact in facts} | {fact.destination.city for fact in facts} == set(
        CITIES
    )
    assert {(fact.delivery_date - fact.pickup_date).days for fact in facts} == {0, 1, 2, 3}
    assert all(fact.pickup_date.strftime("%Y-%m") == fact.load_month for fact in facts)
