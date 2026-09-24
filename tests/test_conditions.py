"""Facts v2: the equipment and conditions table and its stage (sub-step V2, D-083)."""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml
from pydantic import ValidationError

from qf.cli.main import main
from qf.common import QFError, load_artifact_manifest, read_artifact, sha256_file
from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    CargoCategoryV2,
    EquipmentTypeV2,
    LoadFactsV2,
    PackagingType,
    SecuringMethod,
    SensorParameter,
    TemperatureCondition,
)
from qf.data import (
    ConditionsTable,
    build_facts_v2,
    completeness_problems,
    conditions_report,
    generate_dataset,
    load_conditions_table,
    render_conditions_report,
)
from tests.conftest import REPO_ROOT
from tests.generation import REAL_RAW, install_project, repo_config, synthetic_facts

TABLE = REPO_ROOT / "configs" / "data" / "equipment_conditions_v1.yaml"
FACTS = Path("data/processed/load_facts.jsonl")
FACTS_V2 = Path("data/processed/load_facts_v2.jsonl")


def raw_table() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
    return data


def table_with(change: Any) -> ConditionsTable:
    data = copy.deepcopy(raw_table())
    change(data)
    return ConditionsTable.model_validate(data)


def dumps(facts: list[LoadFactsV2]) -> str:
    return "".join(f.model_dump_json() + "\n" for f in facts)


# --- the table -----------------------------------------------------------------------------


def test_committed_table_covers_every_value() -> None:
    """Every equipment type, category, condition kind and list value appears in a profile, so
    the data shows each of them to the model and bench_v2 can test it."""
    table = load_conditions_table(TABLE)
    profiles = [(c, p) for c, items in table.categories.items() for p in items]
    used = {p.equipment for _, p in profiles} | {p.cargo_category or c for c, p in profiles}
    for _, p in profiles:
        used |= {*p.securing, *p.packaging, *p.sensors}
        used |= {k for k, on in (("temperature", p.temperature), ("securing", p.securing),
                                 ("packaging", p.packaging), ("oversize", p.oversize),
                                 ("sensors", p.sensors)) if on}  # fmt: skip
    universe = {
        *get_args(EquipmentTypeV2),
        *get_args(CargoCategoryV2),
        *CARD_V2_CONDITION_KINDS,
        *get_args(SecuringMethod),
        *get_args(PackagingType),
        *get_args(SensorParameter),
    }
    assert universe - used == set()


def _drop_temperature(data: dict[str, Any]) -> None:
    data["categories"]["food_beverage"][0].pop("temperature")


def _lowbed_with_length_only(data: dict[str, Any]) -> None:
    data["categories"]["general"][3]["oversize"] = "long"


@pytest.mark.parametrize(("change", "message"), [
    (lambda d: d["categories"]["retail"][0].update(share=0.5), "shares must sum to 1"),
    (lambda d: d["categories"].pop("retail"), "no profiles for source categories"),
    (lambda d: d["categories"]["retail"][1].update(name="tent"), "profile names repeat"),
    (lambda d: d["categories"]["food_beverage"][0].update(temperature="hot"),
     "unknown temperature range"),
    (lambda d: d["categories"]["general"][3].update(oversize="huge"), "unknown oversize range"),
    (_drop_temperature, "special_conditions.temperature"),
    (_lowbed_with_length_only, "special_conditions.oversize"),
    (lambda d: d["categories"]["retail"][1].update(securing=["load_bars", "load_bars"]),
     "repeats a value"),
    (lambda d: d["categories"]["general"][3].update(pieces=[2, 1]), "pieces must be"),
    (lambda d: d["temperature_ranges"].update(chilled={"min_c": None, "max_c": None}),
     "at least one bound"),
    (lambda d: d["temperature_ranges"].update(chilled={"min_c": 6, "max_c": 2}),
     "must not exceed"),
    (lambda d: d["oversize_ranges"].update(long={"length": None, "width": None, "height": None}),
     "at least one dimension"),
    (lambda d: d["oversize_ranges"].update(long={"length": [16, 13], "width": None,
                                                 "height": None}), "low <= high"),
    (lambda d: d["categories"]["retail"][0].update(equipment="truck"), "equipment"),
    (lambda d: d.update(version="equipment_conditions_v9"), "version"),
])  # fmt: skip
def test_table_is_validated(change: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        table_with(change)


# --- assignment ----------------------------------------------------------------------------


def test_assigned_facts_are_complete_and_follow_the_table() -> None:
    table = load_conditions_table(TABLE)
    source = {f.load_id: f for f in synthetic_facts()}
    facts = build_facts_v2(source.values(), table)
    assert [f.load_id for f in facts] == sorted(source)
    for fact in facts:
        original = source[fact.load_id]
        category, name = fact.profile.split("/")
        assert category == original.cargo_category
        profile = next(p for p in table.categories[original.cargo_category] if p.name == name)
        assert fact.equipment_type == profile.equipment
        assert fact.cargo_category == (profile.cargo_category or original.cargo_category)
        if profile.pieces is None:
            assert fact.pieces == original.pieces
        else:
            assert profile.pieces[0] <= fact.pieces <= profile.pieces[1]
        assert completeness_problems(fact) == []
        kinds = [c.kind for c in fact.special_conditions]
        assert kinds == [k for k in CARD_V2_CONDITION_KINDS if k in kinds]
        for condition in fact.special_conditions:
            if condition.kind == "oversize":
                spans = table.oversize_ranges[profile.oversize or ""]
                for dim, span in (("length", spans.length), ("width", spans.width),
                                  ("height", spans.height)):  # fmt: skip
                    value = getattr(condition, dim)
                    assert (value is None) == (span is None)
                    if value is not None:
                        assert span is not None and span[0] <= value.value <= span[1]
                        assert value.value == round(value.value, 1) and value.unit == "m"
    assert any(f.equipment_type == "lowbed" for f in facts)


def test_an_incomplete_fact_is_reported() -> None:
    fact = build_facts_v2(synthetic_facts()[:50], load_conditions_table(TABLE))
    reefer = next(f for f in fact if f.equipment_type == "reefer")
    bare = reefer.model_copy(update={"special_conditions": []})
    assert completeness_problems(bare) == ["missing special_conditions.temperature"]
    empty = reefer.model_copy(
        update={
            "special_conditions": [TemperatureCondition(kind="temperature", min_c=None, max_c=None)]
        }
    )
    assert completeness_problems(empty) == ["missing special_conditions.temperature"]


def _digest(seed: int | None = None, shuffle: bool = False) -> str:
    table = load_conditions_table(TABLE)
    if seed is not None:
        table = table.model_copy(update={"seed": seed})
    facts = synthetic_facts()
    if shuffle:
        random.Random(3).shuffle(facts)
    return dumps(build_facts_v2(facts, table))


_SCRIPT = """
from pathlib import Path
from qf.data import build_facts_v2, load_conditions_table
from tests.generation import synthetic_facts
facts = build_facts_v2(synthetic_facts(), load_conditions_table(Path({table!r})))
print(__import__("hashlib").sha256("".join(f.model_dump_json() + "\\n" for f in facts).encode())
      .hexdigest())
"""


def test_assignment_is_deterministic() -> None:
    import hashlib

    reference = _digest()
    assert _digest(shuffle=True) == reference
    assert _digest(seed=1) != reference
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT.format(table=str(TABLE))],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
        )  # fmt: skip
        digests.add(result.stdout.strip())
    assert digests == {hashlib.sha256(reference.encode()).hexdigest()}


def test_report_names_every_profile() -> None:
    table = load_conditions_table(TABLE)
    facts = build_facts_v2(synthetic_facts(), table)
    report = conditions_report(facts, table)
    assert report["facts"] == len(facts)
    assert sum(entry["count"] for entry in report["profiles"].values()) == len(facts)
    text = render_conditions_report(report, table)
    for heading in ("## Что проверить", "## Профили по категориям груза из датасета",
                    "## Типы транспорта", "## Значения условий", "## Проверки"):  # fmt: skip
        assert heading in text
    assert "`machinery_lowbed`" in text and "негабарит: длина 6.0–12.0 м" in text
    for wording in ("температура +2…+6 °C", "температура не выше −18 °C", "температура −18 °C",
                    "температура −20…−18 °C"):  # fmt: skip
        assert wording in text


# --- the stage and the command -------------------------------------------------------------


@pytest.fixture
def project(fake_project: Path) -> Path:
    install_project(fake_project)
    shutil.copy(TABLE, fake_project / "configs" / "data" / TABLE.name)
    assert main(["data", "facts"]) == 0
    return fake_project


def test_stage_writes_facts_v2_report_and_lineage(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "facts-v2"]) == 0
    assert "Review: " in capsys.readouterr().out
    v1 = read_artifact(project / FACTS, "load_facts", {"load_facts_v1"}, root=project)
    v2 = read_artifact(project / FACTS_V2, "load_facts", {"load_facts_v2"}, root=project)
    assert v2.parents == (v1.sha256,)
    extra = load_artifact_manifest(project / FACTS_V2, root=project).extra
    assert extra["table"] == "equipment_conditions_v1"
    assert extra["table_sha256"] == sha256_file(TABLE)
    lines = (project / FACTS_V2).read_text(encoding="utf-8").splitlines()
    facts = [LoadFactsV2.model_validate_json(line) for line in lines]
    assert facts and all(completeness_problems(f) == [] for f in facts)
    assert (project / "data/processed/facts_v2_report.md").exists()
    first = v2.sha256
    assert main(["data", "facts-v2"]) == 0  # the same table gives the same bytes
    assert read_artifact(project / FACTS_V2, "load_facts", {"load_facts_v2"},
                         root=project).sha256 == first  # fmt: skip


def test_stage_needs_the_v1_facts(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    install_project(fake_project)
    assert main(["data", "facts-v2", "--table", str(TABLE)]) == 1
    assert "qf data facts" in capsys.readouterr().err


def test_v1_generator_refuses_v2_facts(project: Path) -> None:
    assert main(["data", "facts-v2"]) == 0
    with pytest.raises(QFError, match="unsupported schema version 'load_facts_v2'"):
        generate_dataset(project / FACTS_V2, repo_config(), root=project,
                         out_dir=project / "out", source_prefix="x",
                         config_path=project / "configs/data/generate_v1.yaml")  # fmt: skip


# --- real data -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="raw dataset not fetched")
def test_real_facts_v2() -> None:
    from tests.generation import facts_of

    table = load_conditions_table(TABLE)
    facts = build_facts_v2(facts_of(REAL_RAW), table)
    report = conditions_report(facts, table)
    assert report["facts"] == 85_410
    assert report["unused_values"] == []
    assert report["container_within_limit"]
    by_category: dict[str, int] = {}
    for key, entry in report["profiles"].items():
        by_category[key.split("/")[0]] = by_category.get(key.split("/")[0], 0) + entry["count"]
    for key, entry in report["profiles"].items():
        share = entry["count"] / by_category[key.split("/")[0]]
        assert abs(share - entry["table_share"]) < 0.015, key
    assert json.dumps(report, ensure_ascii=False)
