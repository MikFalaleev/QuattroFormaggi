"""Shared helpers of the generator tests: facts, configs and the invariants E1-E15 (plan 6.9)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qf.common import load_yaml_config, write_artifact
from qf.contracts import LoadFacts, Quantity
from qf.data import (
    GenerateConfig,
    GeneratedRecord,
    build_load_facts,
    load_city_map,
    load_raw_tables,
)
from qf.data.render import MIN_WEIGHT_DIFFERENCE_KG
from qf.domain import (
    CITIES,
    REQUEST_DATE_LABELS,
    canonical_place,
    compute_missing_fields,
    load_system_prompt,
    render_value_from_lbs,
    render_value_per_piece,
    serialize_target,
    to_kg,
)
from tests.conftest import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "raw_mini"
CITY_MAP = REPO_ROOT / "configs" / "data" / "city_map_ru_v1.yaml"
GENERATE_CONFIG = REPO_ROOT / "configs" / "data" / "generate_v1.yaml"
REAL_RAW = REPO_ROOT / "data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07"
SOURCE_PREFIX = "test/dataset@000000000000"


@cache
def facts_of(raw: Path = FIXTURE) -> tuple[LoadFacts, ...]:
    return tuple(build_load_facts(load_raw_tables(raw), load_city_map(CITY_MAP)))


@cache
def money_of(raw: Path = FIXTURE) -> dict[str, set[float]]:
    """Revenue, fuel surcharge and accessorial charges of every load (never in the card)."""
    loads = pd.read_csv(raw / "loads.csv")
    columns = ["revenue", "fuel_surcharge", "accessorial_charges"]
    return {row.load_id: {float(getattr(row, c)) for c in columns} for row in loads.itertuples()}


def repo_config(**changes: Any) -> GenerateConfig:
    config = load_yaml_config(GENERATE_CONFIG, GenerateConfig)
    return config.model_validate({**config.model_dump(), **changes})


def small_config(**changes: Any) -> GenerateConfig:
    """Pools that fit the 10-load fixture: one held-out route, 3 test_ood loads."""
    pool = {"train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2}
    ood = {"holdout_route_count": 1, "holdout_families": ["T7", "T8"]}
    return repo_config(**{"pool": pool, "ood": ood, **changes})


def synthetic_facts(n: int = 600) -> list[LoadFacts]:
    """A varied, deterministic facts set over 12 routes (6 city pairs, both directions)."""
    cities = sorted(CITIES)
    pairs = [(cities[i], cities[(i * 7 + 3) % len(cities)]) for i in range(6)]
    routes = [pair for a, b in pairs for pair in ((a, b), (b, a))]
    categories = ("general", "retail", "consumer_goods", "food_beverage", "automotive",
                  "electronics")  # fmt: skip
    facts = []
    for i in range(n):
        origin, destination = routes[i % len(routes)]
        pickup = date(2022, 1, 1) + timedelta(days=(i * 13) % 1000)
        facts.append(
            LoadFacts(
                load_id=f"LOAD{i:08d}",
                route_id=f"RTE{i % len(routes):05d}",
                customer_id=f"CUST{i % 17:05d}",
                shipper_name=f"Shipper {i % 17}",
                cargo_category=categories[i % 6],
                equipment_type="reefer" if i % 3 == 0 else "dry_van",
                pieces=1 + (i * 5) % 28,
                weight_lbs=10_000 + (i * 7919) % 35_001,
                origin=canonical_place(origin),
                destination=canonical_place(destination),
                pickup_date=pickup,
                delivery_date=pickup + timedelta(days=i % 4),
                load_month=pickup.strftime("%Y-%m"),
            )  # fmt: skip
        )
    return facts


# --- invariants ----------------------------------------------------------------------------

Invariant = Callable[[GeneratedRecord, LoadFacts, set[float]], list[str]]


def _card(g: GeneratedRecord) -> dict[str, Any]:
    return dict(g.rendered.target.card)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, Quantity):
        return [float(value.value)]
    if isinstance(value, dict) and "value" in value:
        return [float(value["value"])]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return []
    return [float(value)]


def _card_numbers(g: GeneratedRecord) -> list[float]:
    target = g.rendered.target
    numbers = [n for value in _card(g).values() for n in _numbers(value)]
    return numbers + [n for c in target.conflicts for v in c.values for n in _numbers(v)]


def e1_nonnull_fields_have_evidence(
    g: GeneratedRecord, f: LoadFacts, money: set[float]
) -> list[str]:
    text, evidence = g.rendered.text, g.rendered.evidence
    problems = [f"{name}: no evidence" for name, value in _card(g).items()
                if value is not None and not evidence.get(name)]  # fmt: skip
    problems += [f"{name}: {s!r} not in text" for name, found in evidence.items() for s in found
                 if s not in text]  # fmt: skip
    return problems


def e2_fields_without_evidence_null(
    g: GeneratedRecord, f: LoadFacts, money: set[float]
) -> list[str]:
    return [f"{name} has no evidence but is {value!r}" for name, value in _card(g).items()
            if name not in g.rendered.evidence and value is not None]  # fmt: skip


def e3_missing_fields_rule(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    target = g.rendered.target
    expected = compute_missing_fields(target.card)
    return [] if target.missing_fields == expected else [f"{target.missing_fields} != {expected}"]


def e4_assistant_canonical(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    content = g.record.messages[2].content
    return [] if content == serialize_target(g.rendered.target) else ["not canonical"]


def e5_no_money_in_card(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    return [f"money value {n} in card" for n in _card_numbers(g) if n in money]


def e6_distractors_not_in_card(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    distractors = {float(v) for v in g.draft.distractors.values()}
    return [f"distractor {n} in card" for n in _card_numbers(g) if n in distractors]


def e7_conflicts_null_and_listed(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    target, card = g.rendered.target, _card(g)
    listed = [c.field for c in target.conflicts]
    problems = [f"{name}: conflict expected" for name in g.draft.conflicts if name not in listed]
    return problems + [
        f"{name}: conflict field not null" for name in listed if card[name] is not None
    ]


def e8_gold_weight_is_rendered(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    card, problems = g.rendered.target.card, []
    if card.weight_total is not None:
        expected = render_value_from_lbs(f.weight_lbs, card.weight_total.unit)
        if card.weight_total.value != expected:
            problems.append(f"weight_total {card.weight_total.value} != rendered {expected}")
    if card.weight_per_piece is not None:
        unit = card.weight_per_piece.unit
        expected = render_value_per_piece(f.weight_lbs, f.pieces, unit)
        if card.weight_per_piece.value != expected:
            problems.append(f"weight_per_piece {card.weight_per_piece.value} != {expected}")
    return problems


def e11_system_prompt(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    same = g.record.messages[0].content == load_system_prompt("system_extract_v1")
    return [] if same else ["system prompt differs"]


def e12_user_starts_with_request_date(
    g: GeneratedRecord, f: LoadFacts, money: set[float]
) -> list[str]:
    label = REQUEST_DATE_LABELS[g.record.language]
    start = f"{label}: {g.draft.request_date:%Y-%m-%d}\n\n"
    return [] if g.record.messages[1].content.startswith(start) else ["no request date line"]


def e13_record_roundtrip(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    same = type(g.record).model_validate_json(g.record.model_dump_json()) == g.record
    return [] if same else ["record does not survive a JSONL round trip"]


def e14_conflict_values_distinct(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    problems = []
    for conflict in g.rendered.target.conflicts:
        mentions = g.rendered.evidence[conflict.field]
        if len(set(mentions)) < 2:
            problems.append(f"{conflict.field}: mentions not distinct {mentions}")
        first, second = conflict.values[:2]
        if conflict.field == "weight_total":
            kg = [to_kg(Quantity.model_validate(v)) for v in (first, second)]
            if abs(kg[0] - kg[1]) <= MIN_WEIGHT_DIFFERENCE_KG:
                problems.append(f"weights too close: {first} {second}")
        elif first == second:
            problems.append(f"{conflict.field}: equal values {first}")
    return problems


def e15_integers_without_decimal(g: GeneratedRecord, f: LoadFacts, money: set[float]) -> list[str]:
    content = g.record.messages[2].content
    return [f"float spelling in {match}" for match in re.findall(r"\d\.0(?!\d)", content)]


INVARIANTS: dict[str, Invariant] = {
    "E1": e1_nonnull_fields_have_evidence,
    "E2": e2_fields_without_evidence_null,
    "E3": e3_missing_fields_rule,
    "E4": e4_assistant_canonical,
    "E5": e5_no_money_in_card,
    "E6": e6_distractors_not_in_card,
    "E7": e7_conflicts_null_and_listed,
    "E8": e8_gold_weight_is_rendered,
    "E11": e11_system_prompt,
    "E12": e12_user_starts_with_request_date,
    "E13": e13_record_roundtrip,
    "E14": e14_conflict_values_distinct,
    "E15": e15_integers_without_decimal,
}


def violations(
    name: str, generated: Sequence[GeneratedRecord], facts: dict[str, LoadFacts],
    money: dict[str, set[float]],
) -> list[str]:  # fmt: skip
    check = INVARIANTS[name]
    found = []
    for g in generated:
        load_id = g.record.group_id.removeprefix("load:")
        found += [
            f"{g.record.id}: {p}" for p in check(g, facts[load_id], money.get(load_id, set()))
        ]
    return found[:10]


# --- a small project on disk ------------------------------------------------------------

RAW_TARGET = Path("data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07")


def install_project(project: Path, **config_changes: Any) -> Path:
    """Raw fixture as an artifact, the data configs and a small generator config."""
    shutil.copytree(FIXTURE, project / RAW_TARGET)
    (project / RAW_TARGET / "provenance.json").write_text(
        (REPO_ROOT / "tests/fixtures/provenance_mini.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_artifact(
        project / RAW_TARGET, "raw_dataset", "logistics_ops_csv_v1", "run-f", root=project
    )
    configs = project / "configs" / "data"
    configs.mkdir(parents=True)
    for name in ("source.yaml", "facts.yaml", "city_map_ru_v1.yaml"):
        shutil.copy(REPO_ROOT / "configs" / "data" / name, configs / name)
    config = small_config(**config_changes).model_dump(mode="json")
    (configs / "generate_v1.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    return configs / "generate_v1.yaml"
