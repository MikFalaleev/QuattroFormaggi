from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from qf.common import load_yaml_config
from qf.data import Expectations
from qf.domain import CITIES, canonical_place
from tests.conftest import REPO_ROOT

FIXTURE_ROUTES = REPO_ROOT / "tests" / "fixtures" / "raw_mini" / "routes.csv"
REAL_ROUTES = (
    REPO_ROOT / "data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07/routes.csv"
)


def _route_places(path: Path) -> set[tuple[str, str]]:
    routes = pd.read_csv(path, dtype=str)
    return {
        (city, state)
        for end in ("origin", "destination")
        for city, state in zip(routes[f"{end}_city"], routes[f"{end}_state"], strict=True)
    }


def _assert_all_known(places: set[tuple[str, str]]) -> None:
    for city, state in places:
        assert canonical_place(city, state).city == city


def test_all_route_cities_present_in_fixture() -> None:
    _assert_all_known(_route_places(FIXTURE_ROUTES))


@pytest.mark.skipif(not REAL_ROUTES.exists(), reason="raw dataset not fetched (qf data fetch)")
def test_all_route_cities_present_in_real_routes() -> None:
    places = _route_places(REAL_ROUTES)
    _assert_all_known(places)
    assert {city for city, _ in places} == set(CITIES)


def test_cities_match_profile_expectations() -> None:
    expectations = load_yaml_config(REPO_ROOT / "configs/data/expectations.yaml", Expectations)
    assert {city: info.state for city, info in CITIES.items()} == dict(expectations.cities)


def test_ru_names_unique() -> None:
    names = [info.name_ru for info in CITIES.values()]
    assert len(names) == len(set(names)) == 20


def test_names_are_well_formed() -> None:
    for city, info in CITIES.items():
        assert re.fullmatch(r"[A-Z]{2}", info.state)
        assert re.fullmatch(r"[А-ЯЁ][а-яё]+(?:[- ][А-ЯЁ][а-яё]+)*", info.name_ru), city
        assert re.fullmatch(r"[А-ЯЁ][а-яё]+(?:[- ][А-ЯЁ][а-яё]+)*", info.state_name_ru), city
        assert re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", info.state_name_en), city


def test_canonical_place_rejects_unknown_city_and_wrong_state() -> None:
    assert canonical_place("Kansas City", "MO").state == "MO"
    with pytest.raises(KeyError, match="unknown city"):
        canonical_place("Канзас-Сити", "MO")
    with pytest.raises(KeyError, match="is in MO, not KS"):
        canonical_place("Kansas City", "KS")


def test_data_spec_city_table_matches_cities() -> None:
    doc = (REPO_ROOT / "docs" / "DATA_SPEC.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| ([A-Z][A-Za-z ]+) \| ([A-Z]{2}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", doc, re.M
    )
    table = {
        city: (state, state_en, city_ru, state_ru)
        for city, state, state_en, city_ru, state_ru in rows
    }
    assert table == {
        city: (info.state, info.state_name_en, info.name_ru, info.state_name_ru)
        for city, info in CITIES.items()
    }
