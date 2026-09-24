"""Facts for card_v2 (sub-step V2, D-083): equipment type and special conditions of each load.

`load_facts_v1` stay unchanged; this stage derives `load_facts_v2` from them with the versioned
table `configs/data/equipment_conditions_v1.yaml`. Each load gets a profile of its source cargo
category, drawn deterministically from the load id, the table version and its seed. Every fact
must yield a fully stated card_v2 card (`qf.domain.compute_missing_fields_v2` is empty), so the
rule "a reefer needs a temperature, a lowbed trailer all three dimensions" lives in one place.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import accumulate
from pathlib import Path
from random import Random
from typing import Any, Final, Literal, get_args

from pydantic import Field, model_validator

from qf.common import (
    ArtifactRef,
    QFError,
    StrictConfig,
    atomic_write_text,
    load_yaml_config,
    read_artifact,
    sha256_file,
    start_run,
    write_artifact,
)
from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    LOAD_FACTS_SCHEMA_VERSION,
    LOAD_FACTS_V2_SCHEMA_VERSION,
    CargoCategory,
    CargoCategoryV2,
    ConditionKind,
    EquipmentTypeV2,
    ExtractionTargetV2,
    Length,
    LoadFacts,
    LoadFactsV2,
    OversizeCondition,
    PackagingCondition,
    PackagingType,
    Place,
    Quantity,
    SecuringCondition,
    SecuringMethod,
    SensorParameter,
    SensorsCondition,
    ShipmentCardV2,
    SpecialCondition,
    TemperatureCondition,
)
from qf.data.facts import load_facts
from qf.domain import LB_TO_KG, check_target_consistency_v2, compute_missing_fields_v2

__all__ = [
    "CONDITIONS_REPORT_VERSION",
    "FACTS_V2_REPORT_FILENAME",
    "LOAD_FACTS_V2_FILENAME",
    "LABELS_RU",
    "ConditionsTable",
    "FactsV2Result",
    "Profile",
    "assign_conditions",
    "build_facts_v2",
    "build_facts_v2_artifact",
    "completeness_problems",
    "conditions_report",
    "describe_conditions",
    "load_conditions_table",
    "render_conditions_report",
]

CONDITIONS_REPORT_VERSION: Final = "facts_v2_report_v1"
LOAD_FACTS_V2_FILENAME: Final = "load_facts_v2.jsonl"
FACTS_V2_REPORT_FILENAME: Final = "facts_v2_report.json"
CONTAINER_MAX_T: Final = 24.0  # payload of a 40 ft container, checked by the report
_RUN_KIND: Final = "facts-v2"
_LIST_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "securing": get_args(SecuringMethod),
    "packaging": get_args(PackagingType),
    "sensors": get_args(SensorParameter),
}
LABELS_RU: Final[dict[str, str]] = {
    "tent": "тент", "van": "фургон", "reefer": "рефрижератор", "isotherm": "изотерм",
    "container": "контейнеровоз", "lowbed": "трал", "mega": "мега", "flatbed": "платформа",
    "general": "генеральный груз", "retail": "розница", "consumer_goods": "ТНП",
    "food_beverage": "продукты", "automotive": "автозапчасти", "electronics": "электроника",
    "machinery": "спецтехника",
    "temperature": "температура", "securing": "крепление", "packaging": "упаковка",
    "oversize": "негабарит", "sensors": "датчики",
    "straps": "ремни", "chains": "цепи", "wheel_chocks": "упоры",
    "anti_slip_mats": "антискользящие коврики", "load_bars": "распорные штанги",
    "crate": "обрешётка", "stretch_film": "стрейч-плёнка",
    "moisture_protection": "влагозащитная", "shock_protection": "амортизирующая",
    "humidity": "влажность", "pressure": "давление", "tilt": "наклон", "shock": "удары",
    "door_opening": "открытие дверей",
}  # fmt: skip
"""Short Russian names for the report (the request wording is the generator's, V3)."""


# --- the table -----------------------------------------------------------------------------


class TemperatureRange(StrictConfig):
    min_c: float | None
    max_c: float | None

    @model_validator(mode="after")
    def _bounds(self) -> TemperatureRange:
        if self.min_c is None and self.max_c is None:
            raise ValueError("a temperature range needs at least one bound")
        if self.min_c is not None and self.max_c is not None and self.min_c > self.max_c:
            raise ValueError("min_c must not exceed max_c")
        return self


Span = tuple[float, float]


class DimensionRanges(StrictConfig):
    """Ranges in metres; null — the dimension is not stated."""

    length: Span | None
    width: Span | None
    height: Span | None

    @model_validator(mode="after")
    def _spans(self) -> DimensionRanges:
        spans = [s for s in (self.length, self.width, self.height) if s is not None]
        if not spans:
            raise ValueError("oversize ranges need at least one dimension")
        if any(not 0 < low <= high for low, high in spans):
            raise ValueError("each range must be [low, high] with 0 < low <= high")
        return self


class Profile(StrictConfig):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    share: float = Field(gt=0, le=1)
    equipment: EquipmentTypeV2
    cargo_category: CargoCategoryV2 | None = None  # the card's category, if not the source's
    pieces: tuple[int, int] | None = None  # [low, high] instead of the source pieces
    temperature: str | None = None  # a key of temperature_ranges
    securing: list[SecuringMethod] = Field(default_factory=list)
    packaging: list[PackagingType] = Field(default_factory=list)
    oversize: str | None = None  # a key of oversize_ranges
    sensors: list[SensorParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _lists(self) -> Profile:
        for name in _LIST_FIELDS:
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{self.name}: {name} repeats a value")
        if self.pieces is not None and not 1 <= self.pieces[0] <= self.pieces[1]:
            raise ValueError(f"{self.name}: pieces must be [low, high] with 1 <= low <= high")
        return self


class ConditionsTable(StrictConfig):
    """`configs/data/equipment_conditions_v1.yaml`."""

    version: Literal["equipment_conditions_v1"]
    seed: int
    temperature_ranges: dict[str, TemperatureRange]
    oversize_ranges: dict[str, DimensionRanges]
    categories: dict[CargoCategory, list[Profile]]

    @model_validator(mode="after")
    def _consistent(self) -> ConditionsTable:
        absent = sorted(set(get_args(CargoCategory)) - set(self.categories))
        if absent:
            raise ValueError(f"no profiles for source categories {absent}")
        for category, profiles in self.categories.items():
            names = [p.name for p in profiles]
            if len(set(names)) != len(names):
                raise ValueError(f"{category}: profile names repeat")
            if abs(sum(p.share for p in profiles) - 1.0) > 1e-9:
                raise ValueError(f"{category}: shares must sum to 1")
            for profile in profiles:
                self._check_profile(category, profile)
        return self

    def _check_profile(self, category: str, profile: Profile) -> None:
        where = f"{category}/{profile.name}"
        if profile.temperature is not None and profile.temperature not in self.temperature_ranges:
            raise ValueError(f"{where}: unknown temperature range {profile.temperature!r}")
        if profile.oversize is not None and profile.oversize not in self.oversize_ranges:
            raise ValueError(f"{where}: unknown oversize range {profile.oversize!r}")
        # the lowest values of the ranges must give a complete card (card_v2 rules)
        probe = self.conditions(profile, _Lowest())
        missing = compute_missing_fields_v2(_probe(profile.equipment, probe))
        if missing:
            raise ValueError(f"{where}: the profile leaves {missing} missing for "
                             f"{profile.equipment}")  # fmt: skip

    def conditions(self, profile: Profile, rng: Random) -> list[SpecialCondition]:
        """The profile's conditions in canonical order; dimensions drawn from the ranges."""
        built: dict[ConditionKind, SpecialCondition] = {}
        if profile.temperature is not None:
            span = self.temperature_ranges[profile.temperature]
            built["temperature"] = TemperatureCondition(kind="temperature", min_c=span.min_c,
                                                        max_c=span.max_c)  # fmt: skip
        if profile.securing:
            built["securing"] = SecuringCondition(
                kind="securing", methods=_ordered(profile.securing, "securing")
            )
        if profile.packaging:
            built["packaging"] = PackagingCondition(
                kind="packaging", types=_ordered(profile.packaging, "packaging")
            )
        if profile.oversize is not None:
            dims = self.oversize_ranges[profile.oversize]
            built["oversize"] = OversizeCondition(
                kind="oversize", length=_draw(dims.length, rng), width=_draw(dims.width, rng),
                height=_draw(dims.height, rng))  # fmt: skip
        if profile.sensors:
            built["sensors"] = SensorsCondition(
                kind="sensors", parameters=_ordered(profile.sensors, "sensors")
            )
        return [built[kind] for kind in CARD_V2_CONDITION_KINDS if kind in built]


class _Lowest(Random):
    """A stand-in RNG that draws the low end of every range (table validation)."""

    def uniform(self, a: float, b: float) -> float:
        return a


def _ordered(values: Sequence[Any], field: str) -> list[Any]:
    return sorted(values, key=_LIST_FIELDS[field].index)


def _draw(span: Span | None, rng: Random) -> Length | None:
    if span is None:
        return None
    return Length(value=round(rng.uniform(*span), 1), unit="m")


def load_conditions_table(path: Path) -> ConditionsTable:
    return load_yaml_config(path, ConditionsTable)


# --- assignment ----------------------------------------------------------------------------


def _card(facts: LoadFactsV2) -> ShipmentCardV2:
    """The fully stated card of the facts, for checking the card_v2 rules."""
    return ShipmentCardV2(
        shipper_name=facts.shipper_name, cargo_category=facts.cargo_category,
        equipment_type=facts.equipment_type, pieces=facts.pieces,
        weight_total=Quantity(value=facts.weight_lbs, unit="lb"), weight_per_piece=None,
        origin=facts.origin, destination=facts.destination, pickup_date=facts.pickup_date,
        delivery_date=facts.delivery_date, special_conditions=facts.special_conditions,
    )  # fmt: skip


def _probe(equipment: EquipmentTypeV2, conditions: list[SpecialCondition]) -> ShipmentCardV2:
    """A card with every field but the conditions stated (table validation)."""
    place = Place(city="Москва", region="Москва")
    return ShipmentCardV2(
        shipper_name="probe", cargo_category="general", equipment_type=equipment, pieces=1,
        weight_total=Quantity(value=1, unit="t"), weight_per_piece=None, origin=place,
        destination=place, pickup_date=date(2024, 1, 1), delivery_date=date(2024, 1, 2),
        special_conditions=conditions,
    )  # fmt: skip


def assign_conditions(facts: LoadFacts, table: ConditionsTable) -> LoadFactsV2:
    """The load's profile and its values. The draws are seeded by the load id, the table version
    and the seed only, so a load gets the same result in any order or process."""
    rng = Random(f"{table.version}:{table.seed}:{facts.load_id}")
    profiles = table.categories[facts.cargo_category]
    bounds = list(accumulate(p.share for p in profiles))
    profile = profiles[min(bisect.bisect_right(bounds, rng.random() * bounds[-1]),
                           len(profiles) - 1)]  # fmt: skip
    conditions = table.conditions(profile, rng)
    pieces = rng.randint(*profile.pieces) if profile.pieces is not None else facts.pieces
    return LoadFactsV2(
        **facts.model_dump(exclude={"cargo_category", "equipment_type", "pieces"}),
        cargo_category=profile.cargo_category or facts.cargo_category,
        equipment_type=profile.equipment, pieces=pieces, special_conditions=conditions,
        profile=f"{facts.cargo_category}/{profile.name}",
    )  # fmt: skip


def completeness_problems(facts: LoadFactsV2) -> list[str]:
    """card_v2 rules violated by the fully stated card of the facts; empty if complete."""
    card = _card(facts)
    target = ExtractionTargetV2(card=card, missing_fields=compute_missing_fields_v2(card),
                                conflicts=[])  # fmt: skip
    problems = [f"missing {name}" for name in target.missing_fields]
    return problems + check_target_consistency_v2(target)


def build_facts_v2(facts: Iterable[LoadFacts], table: ConditionsTable) -> list[LoadFactsV2]:
    """Facts v2 in load_id order; QFError listing the first incomplete facts."""
    built = sorted((assign_conditions(f, table) for f in facts), key=lambda f: f.load_id)
    problems = [f"{f.load_id} ({f.profile}): {p}" for f in built for p in completeness_problems(f)]
    if problems:
        raise QFError(f"{len(problems)} incomplete facts, e.g. {'; '.join(problems[:3])}")
    return built


# --- report --------------------------------------------------------------------------------


def _shares(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {key: {"count": n, "share": round(n / total, 4)} for key, n in counter.most_common()}


def conditions_report(facts: Sequence[LoadFactsV2], table: ConditionsTable) -> dict[str, Any]:
    """Distributions the person checks, the values never used and the container mass check."""
    total = len(facts)
    kinds: Counter[str] = Counter()
    values: Counter[str] = Counter()
    for fact in facts:
        for condition in fact.special_conditions:
            kinds[condition.kind] += 1
            for field in ("methods", "types", "parameters"):
                for value in getattr(condition, field, []):
                    values[f"{condition.kind}.{value}"] += 1
    used = {f.equipment_type for f in facts} | {f.cargo_category for f in facts} | set(kinds) | {
        key.split(".", 1)[1] for key in values}  # fmt: skip
    universe = [*get_args(EquipmentTypeV2), *get_args(CargoCategoryV2), *CARD_V2_CONDITION_KINDS,
                *(v for field in _LIST_FIELDS.values() for v in field)]  # fmt: skip
    container_t = [f.weight_lbs * LB_TO_KG / 1000 for f in facts if f.equipment_type == "container"]
    profiles = Counter(f.profile for f in facts)
    return {
        "table": table.version, "seed": table.seed, "facts": total,
        "with_conditions": {"count": sum(1 for f in facts if f.special_conditions),
                            "share": round(sum(1 for f in facts if f.special_conditions) / total, 4)
                            if total else 0.0},
        "equipment": _shares(Counter(f.equipment_type for f in facts), total),
        "cargo_category": _shares(Counter(f.cargo_category for f in facts), total),
        "condition_kinds": _shares(kinds, total),
        "condition_values": _shares(values, total),
        "profiles": {f"{category}/{p.name}": {"table_share": p.share,
                                              "count": profiles[f"{category}/{p.name}"]}
                     for category, items in table.categories.items() for p in items},
        "unused_values": sorted(set(universe) - used),
        "examples": [_example(next(f for f in facts if f.equipment_type == equipment))
                     for equipment in get_args(EquipmentTypeV2)
                     if any(f.equipment_type == equipment for f in facts)],
        "container_max_t": round(max(container_t), 1) if container_t else None,
        "container_within_limit": all(t <= CONTAINER_MAX_T for t in container_t),
    }  # fmt: skip


def describe_conditions(conditions: Sequence[SpecialCondition]) -> str:
    """«температура +2…+6 °C; датчики: температура» (Russian, for people); a condition named
    without values reads «упаковка: без значений»."""
    parts = []
    for c in conditions:
        match c:
            case TemperatureCondition():
                values = [] if c.min_c is None and c.max_c is None else [
                    _temperature(c.min_c, c.max_c)]  # fmt: skip
                parts.append(_described("температура", values, " "))
            case SecuringCondition():
                parts.append(_described("крепление", [LABELS_RU[v] for v in c.methods]))
            case PackagingCondition():
                parts.append(_described("упаковка", [LABELS_RU[v] for v in c.types]))
            case OversizeCondition():
                units = {"m": "м", "cm": "см"}
                dims = [f"{label} {d.value:g} {units[d.unit]}" for label, d in (
                        ("длина", c.length), ("ширина", c.width), ("высота", c.height))
                        if d is not None]  # fmt: skip
                parts.append(_described("негабарит", dims))
            case SensorsCondition():
                parts.append(_described("датчики", [LABELS_RU[v] for v in c.parameters]))
    return "; ".join(parts) or "без особых условий"


def _described(kind: str, values: Sequence[str], joiner: str = ": ") -> str:
    return f"{kind}{joiner}{', '.join(values)}" if values else f"{kind}: без значений"


def _example(fact: LoadFactsV2) -> dict[str, Any]:
    return {
        "load_id": fact.load_id, "profile": fact.profile,
        "text": (f"{LABELS_RU[fact.equipment_type]}, {LABELS_RU[fact.cargo_category]}, "
                 f"мест: {fact.pieces}, {fact.weight_lbs * LB_TO_KG / 1000:.1f} т, "
                 f"{fact.origin.city} → {fact.destination.city}; "
                 f"{describe_conditions(fact.special_conditions)}"),
    }  # fmt: skip


def _describe(profile: Profile, table: ConditionsTable) -> str:
    parts = []
    if profile.temperature is not None:
        span = table.temperature_ranges[profile.temperature]
        parts.append(f"температура {_temperature(span.min_c, span.max_c)}")
    for name in ("securing", "packaging", "sensors"):
        items = getattr(profile, name)
        if items:
            parts.append(f"{LABELS_RU[name]}: {', '.join(LABELS_RU[v] for v in items)}")
    if profile.oversize is not None:
        dims = table.oversize_ranges[profile.oversize]
        spans = [f"{label} {span[0]}–{span[1]} м" for label, span in
                 (("длина", dims.length), ("ширина", dims.width), ("высота", dims.height))
                 if span is not None]  # fmt: skip
        parts.append("негабарит: " + ", ".join(spans))
    return "; ".join(parts) or "—"


def _celsius(value: float) -> str:
    return f"{value:+g}".replace("-", "\u2212")


def _temperature(min_c: float | None, max_c: float | None) -> str:
    if min_c is None and max_c is not None:
        return f"не выше {_celsius(max_c)} °C"
    if max_c is None and min_c is not None:
        return f"не ниже {_celsius(min_c)} °C"
    assert min_c is not None and max_c is not None
    if min_c == max_c:
        return f"{_celsius(min_c)} °C"
    return f"{_celsius(min_c)}…{_celsius(max_c)} °C"


def render_conditions_report(report: dict[str, Any], table: ConditionsTable) -> str:
    """The Markdown the person reads at the V2 stop."""
    total = report["facts"]
    lines = [
        f"# Транспорт и особые условия: {report['table']}", "",
        f"Загрузок: {total}; seed {report['seed']}. С особыми условиями: "
        f"{report['with_conditions']['count']} ({report['with_conditions']['share']:.0%}).", "",
        "## Что проверить", "",
        "- температурные режимы и какие продукты едут на рефрижераторе, изотерме или тенте;",
        "- габариты спецтехники на трале и длинномеров на платформе;",
        "- какие грузы становятся спецтехникой (часть генеральных грузов) и сколько её;",
        "- долю заявок без особых условий (в плане — около 50%) и долю каждого типа транспорта.",
        "",
        "## Профили по категориям груза из датасета", "",
        "| Категория → профиль | Доля в таблице | Загрузок | Транспорт | Категория в карточке "
        "| Мест | Условия |",
        "|---|---:|---:|---|---|---|---|",
    ]  # fmt: skip
    for category, profiles in table.categories.items():
        for p in profiles:
            count = report["profiles"][f"{category}/{p.name}"]["count"]
            pieces = f"{p.pieces[0]}–{p.pieces[1]}" if p.pieces else "как в датасете"
            target = LABELS_RU[p.cargo_category or category]
            lines.append(f"| {LABELS_RU[category]} → `{p.name}` | {p.share:.0%} | {count} | "
                         f"{LABELS_RU[p.equipment]} | {target} | {pieces} | "
                         f"{_describe(p, table)} |")  # fmt: skip
    for title, key in (("Типы транспорта", "equipment"), ("Категории груза в карточке",
                       "cargo_category"), ("Виды условий", "condition_kinds"),
                       ("Значения условий", "condition_values")):  # fmt: skip
        lines += ["", f"## {title}", "", "| Значение | Загрузок | Доля |", "|---|---:|---:|"]
        for value, entry in report[key].items():
            label = LABELS_RU.get(value.split(".")[-1], value)
            lines.append(f"| `{value}` ({label}) | {entry['count']} | {entry['share']:.1%} |")
    lines += ["", "## Примеры (первая загрузка каждого типа транспорта)", ""]
    lines += [f"- `{e['load_id']}` ({e['profile']}): {e['text']}" for e in report["examples"]]
    unused = ", ".join(f"`{v}`" for v in report["unused_values"]) or "нет"
    lines += ["", "## Проверки", "",
              f"- Значения, которые не встретились ни разу: {unused}.",
              f"- Масса на контейнеровозе: максимум {report['container_max_t']} т "
              f"(предел {CONTAINER_MAX_T:g} т) — "
              f"{'в пределе' if report['container_within_limit'] else 'ПРЕВЫШЕН'}.",
              "- Каждый рефрижератор получил температурный режим, каждый трал — все три "
              "габарита, у каждого условия есть значения (иначе этап останавливается с "
              "ошибкой)."]  # fmt: skip
    return "\n".join(lines) + "\n"


# --- the stage -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FactsV2Result:
    ref: ArtifactRef
    report_path: Path
    count: int
    run_dir: Path


def build_facts_v2_artifact(
    facts_path: Path, table_path: Path, *, root: Path, out_path: Path, report_path: Path
) -> FactsV2Result:
    """`load_facts@load_facts_v1` + table → `load_facts@load_facts_v2`, a report and a run."""
    if not facts_path.exists():
        raise QFError(f"{facts_path} not found; build the v1 facts first: `qf data facts`")
    run = start_run(_RUN_KIND, root)
    source = read_artifact(facts_path, "load_facts", {LOAD_FACTS_SCHEMA_VERSION}, root=root)
    table = load_conditions_table(table_path)
    facts = build_facts_v2(load_facts(source, root), table)
    atomic_write_text(out_path, "".join(fact.model_dump_json() + "\n" for fact in facts))
    table_sha = sha256_file(table_path)
    ref = write_artifact(
        out_path, "load_facts", LOAD_FACTS_V2_SCHEMA_VERSION, run.run_id, [source.sha256],
        extra={"count": len(facts), "table": table.version, "table_sha256": table_sha,
               "seed": table.seed}, root=root,
    )  # fmt: skip
    report = conditions_report(facts, table)
    atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report_ref = write_artifact(report_path, "metrics", CONDITIONS_REPORT_VERSION, run.run_id,
                                [ref.sha256], root=root)  # fmt: skip
    atomic_write_text(report_path.with_suffix(".md"), render_conditions_report(report, table))
    relative_table = table_path.resolve().relative_to(root.resolve()).as_posix() if (
        table_path.resolve().is_relative_to(root.resolve())) else str(table_path)  # fmt: skip
    run.finish(
        config={"table": relative_table, "version": table.version, "seed": table.seed},
        data_hashes={source.path.as_posix(): source.sha256, relative_table: table_sha,
                     ref.path.as_posix(): ref.sha256,
                     report_ref.path.as_posix(): report_ref.sha256},
        metrics={"load_facts_v2": len(facts), "with_conditions": report["with_conditions"]},
    )  # fmt: skip
    return FactsV2Result(ref=ref, report_path=report_path.with_suffix(".md"), count=len(facts),
                         run_dir=run.run_dir)  # fmt: skip
