from __future__ import annotations

import math
import statistics
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from qf.common import QFError, load_yaml_config
from qf.contracts import Place
from qf.data import CityMap, Expectations, load_city_map
from qf.domain import CITIES
from tests.conftest import REPO_ROOT

CITY_MAP = REPO_ROOT / "configs" / "data" / "city_map_ru_v1.yaml"
FIXTURE_ROUTES = REPO_ROOT / "tests" / "fixtures" / "raw_mini" / "routes.csv"
REAL_ROUTES = (
    REPO_ROOT / "data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07/routes.csv"
)
ROAD_FACTOR = 1.2  # road km per great-circle km, rough average for Russia (D-047)


def _routes(path: Path) -> list[tuple[str, str, float]]:
    routes = pd.read_csv(path)
    return list(
        zip(routes.origin_city, routes.destination_city, routes.typical_distance_miles, strict=True)
    )


def _great_circle_km(a: str, b: str) -> float:
    p1, p2 = math.radians(CITIES[a].lat), math.radians(CITIES[b].lat)
    dlon = math.radians(CITIES[b].lon - CITIES[a].lon)
    cos = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dlon)
    return 6371.0 * math.acos(min(1.0, cos))


def test_map_is_one_to_one_from_source_cities_onto_catalog() -> None:
    city_map = load_city_map(CITY_MAP)
    expectations = load_yaml_config(REPO_ROOT / "configs/data/expectations.yaml", Expectations)
    assert set(city_map.cities) == set(expectations.cities)
    assert sorted(city_map.cities.values()) == sorted(CITIES)


def test_every_fixture_route_gets_two_different_russian_cities() -> None:
    city_map = load_city_map(CITY_MAP)
    for origin, destination, _ in _routes(FIXTURE_ROUTES):
        assert city_map.place(origin) != city_map.place(destination)
    assert city_map.place("Chicago") == Place(city="Казань", region="Республика Татарстан")


@pytest.mark.skipif(not REAL_ROUTES.exists(), reason="raw dataset not fetched (qf data fetch)")
def test_russian_distances_follow_source_distances() -> None:
    """Transit days of the source grow with distance, so mapped routes must keep their length."""
    city_map = load_city_map(CITY_MAP)
    source_km, russian_km = [], []
    for origin, destination, miles in _routes(REAL_ROUTES):
        a, b = city_map.cities[origin], city_map.cities[destination]
        assert a != b
        source_km.append(miles * 1.609344)
        russian_km.append(_great_circle_km(a, b) * ROAD_FACTOR)
    assert len(source_km) == 58
    assert statistics.correlation(source_km, russian_km) > 0.9
    deviations = [abs(r / s - 1) for s, r in zip(source_km, russian_km, strict=True)]
    assert statistics.median(deviations) < 0.2
    assert max(deviations) < 0.5


def test_map_rejects_unknown_and_shared_cities() -> None:
    with pytest.raises(ValidationError, match="not in qf.domain.CITIES"):
        CityMap(version="city_map_ru_v1", cities={"Chicago": "Kazan"})
    with pytest.raises(ValidationError, match="several source cities map to"):
        CityMap(version="city_map_ru_v1", cities={"Chicago": "Казань", "Dallas": "Казань"})
    with pytest.raises(ValidationError):
        CityMap(version="city_map_ru_v2", cities={})  # type: ignore[arg-type]


def test_unknown_source_city_is_explicit_error() -> None:
    with pytest.raises(QFError, match="'Boston' is not in city map city_map_ru_v1"):
        load_city_map(CITY_MAP).place("Boston")
