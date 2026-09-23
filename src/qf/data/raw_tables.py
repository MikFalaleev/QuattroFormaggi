"""The four raw CSV tables used by the pipeline, loaded with explicit columns and dtypes (step 3).

Identifiers and categories are strings; counts are nullable integers so that a missing value is
reported by the null-rate check instead of crashing the loader; dates are parsed with explicit
formats and a value that does not match is an error naming the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qf.common import DataValidationError, Issue, QFError

__all__ = [
    "DATE_FORMATS",
    "DTYPES",
    "EXPECTED_COLUMNS",
    "PRIMARY_KEYS",
    "RawTables",
    "load_raw_tables",
]

DTYPES: dict[str, dict[str, str]] = {
    "loads": {
        "load_id": "str",
        "customer_id": "str",
        "route_id": "str",
        "load_date": "str",
        "load_type": "str",
        "weight_lbs": "Int64",
        "pieces": "Int64",
        "revenue": "Float64",
        "fuel_surcharge": "Float64",
        "accessorial_charges": "Float64",
        "load_status": "str",
        "booking_type": "str",
    },
    "routes": {
        "route_id": "str",
        "origin_city": "str",
        "origin_state": "str",
        "destination_city": "str",
        "destination_state": "str",
        "typical_distance_miles": "Int64",
        "base_rate_per_mile": "Float64",
        "fuel_surcharge_rate": "Float64",
        "typical_transit_days": "Int64",
    },
    "customers": {
        "customer_id": "str",
        "customer_name": "str",
        "customer_type": "str",
        "credit_terms_days": "Int64",
        "primary_freight_type": "str",
        "account_status": "str",
        "contract_start_date": "str",
        "annual_revenue_potential": "Int64",
    },
    "delivery_events": {
        "event_id": "str",
        "load_id": "str",
        "trip_id": "str",
        "event_type": "str",
        "facility_id": "str",
        "scheduled_datetime": "str",
        "actual_datetime": "str",
        "detention_minutes": "Int64",
        "on_time_flag": "str",
        "location_city": "str",
        "location_state": "str",
    },
}
EXPECTED_COLUMNS: dict[str, list[str]] = {table: list(dtypes) for table, dtypes in DTYPES.items()}
PRIMARY_KEYS: dict[str, str] = {
    "loads": "load_id",
    "routes": "route_id",
    "customers": "customer_id",
    "delivery_events": "event_id",
}
# Only the date columns the pipeline uses are parsed; the rest stay strings.
DATE_FORMATS: dict[str, dict[str, str]] = {
    "loads": {"load_date": "%Y-%m-%d"},
    "delivery_events": {"scheduled_datetime": "%Y-%m-%d %H:%M:%S.%f"},
}


@dataclass(frozen=True)
class RawTables:
    loads: pd.DataFrame
    routes: pd.DataFrame
    customers: pd.DataFrame
    delivery_events: pd.DataFrame

    def row_counts(self) -> dict[str, int]:
        return {table: len(getattr(self, table)) for table in DTYPES}


def _check_columns(path: Path, table: str) -> None:
    header = list(pd.read_csv(path, nrows=0).columns)
    expected = EXPECTED_COLUMNS[table]
    missing = [column for column in expected if column not in header]
    unexpected = [column for column in header if column not in expected]
    if missing or unexpected:
        message = (
            f"columns differ from the expected set: missing {missing}, unexpected {unexpected}"
        )
        raise DataValidationError([Issue(path.name, None, "COLUMNS", message)])


def _parse_dates(frame: pd.DataFrame, table: str, file_name: str) -> pd.DataFrame:
    for column, date_format in DATE_FORMATS.get(table, {}).items():
        raw = frame[column]
        parsed = pd.to_datetime(raw, format=date_format, errors="coerce")
        bad = raw.notna() & parsed.isna()
        if bad.any():
            key = PRIMARY_KEYS[table]
            raise DataValidationError(
                [
                    Issue(str(record_id), None, "DATE_FORMAT", f"{file_name}: {column}={value!r}")
                    for record_id, value in zip(frame.loc[bad, key], raw[bad], strict=True)
                ]
            )
        frame[column] = parsed
    return frame


def _read_table(raw_dir: Path, table: str) -> pd.DataFrame:
    path = raw_dir / f"{table}.csv"
    if not path.is_file():
        raise QFError(f"raw table {path} is missing")
    _check_columns(path, table)
    try:
        frame = pd.read_csv(path, dtype=DTYPES[table])
    except (ValueError, TypeError) as exc:
        raise DataValidationError([Issue(path.name, None, "DTYPE", str(exc))]) from exc
    return _parse_dates(frame, table, path.name)


def load_raw_tables(raw_dir: Path) -> RawTables:
    """Load loads, routes, customers and delivery_events from a fetched raw dataset directory."""
    return RawTables(**{table: _read_table(raw_dir, table) for table in DTYPES})
