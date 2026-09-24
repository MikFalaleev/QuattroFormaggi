"""Shared helpers of the generator tests: facts, configs and the invariants E1-E17 (plan 6.9;
E16-E17 check the special conditions of card_v2, sub-step V3)."""

from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qf.common import load_yaml_config, write_artifact
from qf.contracts import (
    AnyLoadFacts,
    LoadFacts,
    LoadFactsV2,
    OversizeCondition,
    PackagingCondition,
    Quantity,
    SecuringCondition,
    SensorsCondition,
    TemperatureCondition,
)
from qf.data import (
    GenerateConfig,
    GeneratedRecord,
    build_facts_v2,
    build_load_facts,
    load_city_map,
    load_conditions_table,
    load_raw_tables,
)
from qf.data.render import MIN_WEIGHT_DIFFERENCE_KG
from qf.data.render.vocabulary import (
    EN_PACKAGING,
    EN_SECURING,
    EN_SENSORS,
    RU_PACKAGING,
    RU_SECURING,
    RU_SENSORS,
)
from qf.domain import (
    CITIES,
    REQUEST_DATE_LABELS,
    canonical_place,
    get_target_schema,
    get_task,
    load_system_prompt,
    render_value_from_lbs,
    render_value_per_piece,
    serialize_target,
    to_kg,
    to_m,
)
from tests.conftest import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "raw_mini"
CITY_MAP = REPO_ROOT / "configs" / "data" / "city_map_ru_v1.yaml"
GENERATE_CONFIG = REPO_ROOT / "configs" / "data" / "generate_v1.yaml"
GENERATE_V2_CONFIG = REPO_ROOT / "configs" / "data" / "generate_v2.yaml"
CONDITIONS_TABLE = REPO_ROOT / "configs" / "data" / "equipment_conditions_v1.yaml"
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


def repo_config_v2(**changes: Any) -> GenerateConfig:
    config = load_yaml_config(GENERATE_V2_CONFIG, GenerateConfig)
    return config.model_validate({**config.model_dump(), **changes})


@cache
def synthetic_facts_v2() -> tuple[LoadFactsV2, ...]:
    """`synthetic_facts` through the committed equipment and conditions table."""
    return tuple(build_facts_v2(synthetic_facts(), load_conditions_table(CONDITIONS_TABLE)))


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

Invariant = Callable[[GeneratedRecord, AnyLoadFacts, set[float]], list[str]]


def _card(g: GeneratedRecord) -> dict[str, Any]:
    return dict(g.rendered.target.card)


def _stated(value: Any) -> bool:
    """A card value that the text must state: not null and not an empty condition list."""
    return value is not None and value != []


def _numbers(value: Any) -> list[float]:
    if isinstance(value, Quantity):
        return [float(value.value)]
    if isinstance(value, dict) and "value" in value:
        return [float(value["value"])]
    if isinstance(value, list):  # card_v2 conditions: temperatures and dimensions
        return [n for item in value for n in _condition_numbers(item)]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return []
    return [float(value)]


def _condition_numbers(condition: Any) -> list[float]:
    if isinstance(condition, TemperatureCondition):
        return [abs(v) for v in (condition.min_c, condition.max_c) if v is not None]
    if isinstance(condition, OversizeCondition):
        return [float(d.value) for d in (condition.length, condition.width, condition.height)
                if d is not None]  # fmt: skip
    return []


def _card_numbers(g: GeneratedRecord, *, conditions: bool = True) -> list[float]:
    """Numbers of the card and its conflicts; `conditions=False` leaves out the temperatures
    and dimensions (they come from the conditions table, not from the source)."""
    target = g.rendered.target
    numbers = [n for name, value in _card(g).items() if conditions or name != "special_conditions"
               for n in _numbers(value)]  # fmt: skip
    return numbers + [n for c in target.conflicts for v in c.values for n in _numbers(v)]


def e1_nonnull_fields_have_evidence(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    text, evidence = g.rendered.text, g.rendered.evidence
    problems = [f"{name}: no evidence" for name, value in _card(g).items()
                if _stated(value) and not evidence.get(name)]  # fmt: skip
    problems += [f"{name}: {s!r} not in text" for name, found in evidence.items() for s in found
                 if s not in text]  # fmt: skip
    return problems


def e2_fields_without_evidence_null(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    return [f"{name} has no evidence but is {value!r}" for name, value in _card(g).items()
            if name not in g.rendered.evidence and _stated(value)]  # fmt: skip


def e3_missing_fields_rule(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    target = g.rendered.target
    expected = get_target_schema(g.record.schema_version).missing_fields(target.card)
    return [] if target.missing_fields == expected else [f"{target.missing_fields} != {expected}"]


def e4_assistant_canonical(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    content = g.record.messages[2].content
    return [] if content == serialize_target(g.rendered.target) else ["not canonical"]


def _source_numbers(g: GeneratedRecord, f: AnyLoadFacts) -> set[float]:
    """Numbers the card may legitimately hold: pieces and weights rendered from the facts and
    from the draft's conflict values, in every unit."""
    numbers = {float(f.pieces), *(float(v) for v in g.draft.conflicts.values())}
    for lbs in (f.weight_lbs, g.draft.conflicts.get("weight_total")):
        if lbs is None:
            continue
        for unit in ("kg", "t", "lb"):
            numbers.add(float(render_value_from_lbs(lbs, unit)))
            with contextlib.suppress(ValueError):  # a piece may round to 0 t
                numbers.add(float(render_value_per_piece(lbs, f.pieces, unit)))
    return numbers


def e5_no_money_in_card(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    """No card number comes from revenue or charges. A number that equals a money value only
    by coincidence (13.8 t of weight next to a 13.8 fuel surcharge) is explained by the source
    weight or pieces; conditions come from the conditions table and are not compared."""
    explained = _source_numbers(g, f)
    return [f"money value {n} in card" for n in _card_numbers(g, conditions=False)
            if n in money and n not in explained]  # fmt: skip


def e6_distractors_not_in_card(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    distractors = {float(v) for v in g.draft.distractors.values()}
    return [f"distractor {n} in card" for n in _card_numbers(g) if n in distractors]


def e7_conflicts_null_and_listed(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    target, card = g.rendered.target, _card(g)
    listed = [c.field for c in target.conflicts]
    problems = [f"{name}: conflict expected" for name in g.draft.conflicts if name not in listed]
    return problems + [
        f"{name}: conflict field not null" for name in listed if card[name] is not None
    ]


def e8_gold_weight_is_rendered(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
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


def e11_system_prompt(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    version = get_task(g.record.task).prompt_for(g.record.schema_version)
    same = g.record.messages[0].content == load_system_prompt(version)
    return [] if same else ["system prompt differs"]


def e12_user_starts_with_request_date(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    label = REQUEST_DATE_LABELS[g.record.language]
    start = f"{label}: {g.draft.request_date:%Y-%m-%d}\n\n"
    return [] if g.record.messages[1].content.startswith(start) else ["no request date line"]


def e13_record_roundtrip(g: GeneratedRecord, f: AnyLoadFacts, money: set[float]) -> list[str]:
    same = type(g.record).model_validate_json(g.record.model_dump_json()) == g.record
    return [] if same else ["record does not survive a JSONL round trip"]


def e14_conflict_values_distinct(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
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


def e15_integers_without_decimal(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    content = g.record.messages[2].content
    return [f"float spelling in {match}" for match in re.findall(r"\d\.0(?!\d)", content)]


def _expected_conditions(f: AnyLoadFacts, g: GeneratedRecord) -> list[Any]:
    """The conditions the gold must hold, from the facts and the draft alone: dropped kinds
    absent, kinds without values empty, oversize with the kept dimensions only."""
    if not isinstance(f, LoadFactsV2):
        return []
    draft, expected = g.draft, []
    for c in f.special_conditions:
        if c.kind in draft.conditions_dropped:
            continue
        if c.kind in draft.conditions_without_values:
            expected.append((c.kind, None))
        elif isinstance(c, OversizeCondition):
            kept = draft.oversize_dims or ["length", "width", "height"]
            expected.append((c.kind, {d: to_m(getattr(c, d)) for d in kept
                                      if getattr(c, d) is not None}))  # fmt: skip
        else:
            expected.append((c.kind, c.model_dump(exclude={"kind"})))
    return expected


def _values_of(condition: Any) -> Any:
    empty = {"temperature": {"min_c": None, "max_c": None}, "securing": {"methods": []},
             "packaging": {"types": []}, "sensors": {"parameters": []}}  # fmt: skip
    if isinstance(condition, OversizeCondition):
        dims = {d: to_m(getattr(condition, d)) for d in ("length", "width", "height")
                if getattr(condition, d) is not None}  # fmt: skip
        return dims or None
    values = condition.model_dump(exclude={"kind"})
    return None if values == empty[condition.kind] else values


def e16_conditions_follow_facts_and_draft(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    gold = [(c.kind, _values_of(c)) for c in getattr(g.rendered.target.card,
                                                     "special_conditions", [])]  # fmt: skip
    expected = _expected_conditions(f, g)
    if [k for k, _ in gold] != [k for k, _ in expected]:
        return [f"condition kinds {[k for k, _ in gold]} != {[k for k, _ in expected]}"]
    problems = []
    for (kind, got), (_, want) in zip(gold, expected, strict=True):
        if kind == "oversize" and got is not None and want is not None:
            if set(got) != set(want) or any(abs(got[d] - want[d]) > 0.01 for d in want):
                problems.append(f"oversize {got} != {want}")
        elif got != want:
            problems.append(f"{kind} {got} != {want}")
    return problems


def _tokens(condition: Any, lang: str) -> list[tuple[str, ...]]:
    """Alternatives of text that must be in the evidence of the condition, per value."""
    ru = lang == "ru"
    if isinstance(condition, TemperatureCondition):
        return [(_digits(abs(v), lang),) for v in (condition.min_c, condition.max_c)
                if v is not None]  # fmt: skip
    if isinstance(condition, OversizeCondition):
        return [(_digits(d.value, lang),) for d in (condition.length, condition.width,
                                                  condition.height) if d is not None]  # fmt: skip
    if isinstance(condition, SecuringCondition):
        return [RU_SECURING[m] if ru else (EN_SECURING[m],) for m in condition.methods]
    if isinstance(condition, PackagingCondition):
        return [RU_PACKAGING[t] if ru else (EN_PACKAGING[t],) for t in condition.types]
    if isinstance(condition, SensorsCondition):
        logger = ("термописец", "термописцем") if ru else ("temperature logger",)
        return [(RU_SENSORS[p] if ru else EN_SENSORS[p],) + (logger if p == "temperature" else ())
                for p in condition.parameters]  # fmt: skip
    return []


def _digits(value: float, lang: str) -> str:
    text = str(int(value)) if float(value).is_integer() else f"{value:.1f}"
    return text.replace(".", ",") if lang == "ru" else text


def e17_condition_values_in_evidence(
    g: GeneratedRecord, f: AnyLoadFacts, money: set[float]
) -> list[str]:
    conditions = getattr(g.rendered.target.card, "special_conditions", [])
    mentions = g.rendered.evidence.get("special_conditions", [])
    if len(mentions) != len(conditions):
        return [f"{len(conditions)} conditions, {len(mentions)} condition mentions"]
    problems = []
    for condition in conditions:
        needed = _tokens(condition, g.record.language)
        lowered = [m.lower() for m in mentions]  # a list line may start with a capital
        if not any(all(any(t.lower() in m for t in alts) for alts in needed) for m in lowered):
            problems.append(f"{condition.kind}: values {needed} not in {mentions}")
    return problems


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
    "E16": e16_conditions_follow_facts_and_draft,
    "E17": e17_condition_values_in_evidence,
}


def violations(
    name: str, generated: Sequence[GeneratedRecord], facts: Mapping[str, AnyLoadFacts],
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
