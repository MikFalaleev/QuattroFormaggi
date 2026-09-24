"""Load facts (step 5): the logistics-operations builder, the stage and JSONL storage.

The stage depends only on the `FactsBuilder` port; `LogisticsOpsFactsBuilder` is the
implementation for this dataset. Cities are replaced by Russian ones here (D-047).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from qf.common import (
    ArtifactRef,
    ComponentConfig,
    DataValidationError,
    Issue,
    QFError,
    StrictConfig,
    atomic_write_text,
    format_validation_error,
    read_artifact,
    sha256_file,
    start_run,
    write_artifact,
)
from qf.contracts import (
    LOAD_FACTS_SCHEMA_VERSION,
    LOAD_FACTS_V2_SCHEMA_VERSION,
    CargoCategory,
    EquipmentType,
    FactsBuilder,
    LoadFacts,
    LoadFactsV2,
    Place,
    supported_versions,
)
from qf.data.city_map import CityMap, load_city_map
from qf.data.fetch import RAW_SCHEMA_VERSION
from qf.data.raw_tables import RawTables, load_raw_tables, pickup_delivery_by_load
from qf.data.registries import FACTS_BUILDERS

__all__ = [
    "CARGO_CATEGORIES",
    "EQUIPMENT_TYPES",
    "FACTS_MODELS",
    "LOAD_FACTS_FILENAME",
    "FactsFileConfig",
    "FactsResult",
    "LogisticsOpsFactsBuilder",
    "build_facts_artifact",
    "build_load_facts",
    "load_facts",
    "save_facts",
]

LOAD_FACTS_FILENAME: Final = "load_facts.jsonl"
FACTS_MODELS: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {LOAD_FACTS_SCHEMA_VERSION: LoadFacts, LOAD_FACTS_V2_SCHEMA_VERSION: LoadFactsV2}
)
"""The model of each `load_facts` version (both carry load_id and route_id)."""
EQUIPMENT_TYPES: Final[Mapping[str, EquipmentType]] = MappingProxyType(
    {"Dry Van": "dry_van", "Refrigerated": "reefer"}
)
CARGO_CATEGORIES: Final[Mapping[str, CargoCategory]] = MappingProxyType(
    {
        "General": "general",
        "Retail": "retail",
        "Consumer Goods": "consumer_goods",
        "Food/Beverage": "food_beverage",
        "Automotive": "automotive",
        "Electronics": "electronics",
    }
)
_RUN_KIND = "facts"
_REQUIRED_COLUMNS = ("route_id", "customer_id", "load_date", "load_type", "weight_lbs", "pieces")


class FactsFileConfig(StrictConfig):
    """`configs/data/facts.yaml`: which builder makes the facts, with its parameters."""

    builder: ComponentConfig


class _RowError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _merge_tables(t: RawTables) -> pd.DataFrame:
    try:
        return (
            t.loads.merge(t.routes[["route_id", "origin_city", "destination_city"]],
                          on="route_id", how="left", validate="many_to_one")
            .merge(t.customers[["customer_id", "customer_name", "primary_freight_type"]],
                   on="customer_id", how="left", validate="many_to_one")
            .merge(pickup_delivery_by_load(t), on="load_id", how="left", validate="one_to_one")
            .sort_values("load_id", kind="stable")
        )  # fmt: skip
    except pd.errors.MergeError as exc:
        raise QFError(f"raw tables have duplicate keys (run `qf data profile`): {exc}") from exc


def _lookup(mapping: Mapping[str, Any], value: Any, code: str, what: str) -> Any:
    if value not in mapping:
        raise _RowError(code, f"{what} {value!r} is not known")
    return mapping[value]


def _facts_of(row: Any, places: Mapping[str, Place]) -> LoadFacts:
    missing = [column for column in _REQUIRED_COLUMNS if pd.isna(getattr(row, column))]
    if missing:
        raise _RowError("MISSING_VALUE", f"empty {', '.join(missing)}")
    if pd.isna(row.origin_city):
        raise _RowError("MISSING_ROUTE", f"route {row.route_id} is not in routes")
    if pd.isna(row.customer_name):
        raise _RowError("MISSING_CUSTOMER", f"customer {row.customer_id} is not in customers")
    if pd.isna(row.pickup_scheduled_datetime):
        raise _RowError("EVENTS", "needs exactly one Pickup and one Delivery event")
    try:
        return LoadFacts(
            load_id=str(row.load_id),
            route_id=str(row.route_id),
            customer_id=str(row.customer_id),
            shipper_name=str(row.customer_name),
            cargo_category=_lookup(CARGO_CATEGORIES, row.primary_freight_type, "UNKNOWN_VALUE",
                                   "primary_freight_type"),
            equipment_type=_lookup(EQUIPMENT_TYPES, row.load_type, "UNKNOWN_VALUE", "load_type"),
            pieces=int(row.pieces),
            weight_lbs=int(row.weight_lbs),
            origin=_lookup(places, row.origin_city, "UNKNOWN_CITY", "origin city"),
            destination=_lookup(places, row.destination_city, "UNKNOWN_CITY", "destination city"),
            pickup_date=row.pickup_scheduled_datetime.date(),
            delivery_date=row.delivery_scheduled_datetime.date(),
            load_month=row.load_date.strftime("%Y-%m"),
        )  # fmt: skip
    except ValidationError as exc:
        raise _RowError("INVALID", format_validation_error(exc)) from exc


def build_load_facts(t: RawTables, city_map: CityMap) -> list[LoadFacts]:
    """One LoadFacts per load, sorted by load_id; every bad load is reported at once.

    The route comes from `routes` (never from facilities) with its cities replaced by
    `city_map`; dates are the calendar dates of the Pickup and Delivery events.
    """
    places = {city: city_map.place(city) for city in city_map.cities}  # 20 shared objects
    facts, issues = [], []
    for row in _merge_tables(t).itertuples(index=False):
        try:
            facts.append(_facts_of(row, places))
        except _RowError as exc:
            issues.append(Issue(str(row.load_id), None, exc.code, exc.message))
    if issues:
        raise DataValidationError(issues)
    return facts


@FACTS_BUILDERS.register("logistics_operations_v1")
class LogisticsOpsFactsBuilder:
    """`FactsBuilder` for yogape/logistics-operations (raw format `logistics_ops_csv_v1`)."""

    class Config(StrictConfig):
        city_map: Path  # relative to the project root

        @field_validator("city_map")
        @classmethod
        def _relative(cls, value: Path) -> Path:
            if value.is_absolute():
                raise ValueError("city_map must be relative to the project root")
            return value

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(self, raw: ArtifactRef, root: Path) -> list[LoadFacts]:
        if raw.schema_version != RAW_SCHEMA_VERSION:
            raise QFError(f"{raw.path}: cannot read raw format '{raw.schema_version}'")
        city_map = load_city_map(root / self.config.city_map)
        return build_load_facts(load_raw_tables(root / raw.path), city_map)

    def inputs(self, root: Path) -> dict[str, str]:
        return {self.config.city_map.as_posix(): sha256_file(root / self.config.city_map)}


def save_facts(
    facts: Sequence[LoadFacts],
    path: Path,
    *,
    run_id: str,
    parents: Sequence[str],
    root: Path,
    extra: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Write JSONL (one record per line, in the given order) and its artifact manifest."""
    atomic_write_text(path, "".join(fact.model_dump_json() + "\n" for fact in facts))
    return write_artifact(
        path, "load_facts", LOAD_FACTS_SCHEMA_VERSION, run_id, parents, extra=extra, root=root
    )


def load_facts(ref: ArtifactRef, root: Path, *, model: type[BaseModel] = LoadFacts) -> list[Any]:
    """Records of a `load_facts` artifact already verified by `read_artifact`, parsed with
    `model` (`LoadFacts`, or `LoadFactsV2` for `load_facts_v2`)."""
    path = root / ref.path
    facts = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            facts.append(model.model_validate_json(line))
        except ValidationError as exc:
            issue = Issue(f"{ref.path}:{number}", None, "SCHEMA", format_validation_error(exc))
            raise DataValidationError([issue]) from exc
    return facts


@dataclass(frozen=True)
class FactsResult:
    ref: ArtifactRef
    count: int
    run_dir: Path


def build_facts_artifact(
    builder: FactsBuilder,
    raw_path: Path,
    *,
    root: Path,
    out_path: Path,
    config: Mapping[str, Any],
) -> FactsResult:
    """The facts stage: verified raw artifact -> `load_facts@load_facts_v1` + run manifest."""
    run = start_run(_RUN_KIND, root)
    raw_ref = read_artifact(raw_path, "raw_dataset", supported_versions("raw_dataset"), root=root)
    inputs = builder.inputs(root)
    facts = sorted(builder.build(raw_ref, root), key=lambda fact: fact.load_id)
    duplicates = sorted(i for i, n in Counter(fact.load_id for fact in facts).items() if n > 1)
    if duplicates:
        raise QFError(f"duplicate load_id in facts: {', '.join(duplicates[:5])}")
    ref = save_facts(
        facts, out_path, run_id=run.run_id, parents=[raw_ref.sha256], root=root,
        extra={"count": len(facts), "inputs": inputs},
    )  # fmt: skip
    run.finish(
        packages=["pandas"],
        data_hashes={raw_ref.path.as_posix(): raw_ref.sha256, **inputs,
                     ref.path.as_posix(): ref.sha256},
        config=dict(config),
        metrics={"load_facts": len(facts)},
    )  # fmt: skip
    return FactsResult(ref=ref, count=len(facts), run_dir=run.run_dir)
