from __future__ import annotations

import re

import pytest

from qf.contracts import Place
from qf.domain import CITIES, canonical_place
from tests.conftest import REPO_ROOT

CYRILLIC_NAME = r"[А-ЯЁ][а-яё]+(?:[- ](?:на-)?[А-ЯЁа-яё][а-яё]+)*"


def test_catalog_has_20_cities_with_unique_names() -> None:
    assert len(CITIES) == 20
    for field in ("name_en", "genitive"):
        values = [getattr(info, field) for info in CITIES.values()]
        assert len(set(values)) == 20, field
    assert {"Москва", "Санкт-Петербург", "Казань", "Новосибирск"} <= set(CITIES)


def test_names_are_well_formed() -> None:
    for city, info in CITIES.items():
        assert re.fullmatch(CYRILLIC_NAME, city), city
        assert re.fullmatch(r"[A-Z][A-Za-z -]+", info.name_en), city
        assert re.fullmatch(r"[A-Z][A-Za-z ]+", info.region_en), city
        for form in (info.genitive, info.accusative):
            assert re.fullmatch(CYRILLIC_NAME, form), city
            assert form[:2] == city[:2], city


def test_region_is_an_official_subject_name() -> None:
    for city, info in CITIES.items():
        federal_city = info.region == city
        assert federal_city or re.fullmatch(
            r"Республика [А-ЯЁ][а-яё]+|[А-ЯЁ][а-яё]+ (?:область|край)(?: — [А-ЯЁ][а-яё]+)?",
            info.region,
        ), city
    assert [c for c, info in CITIES.items() if info.region == c] == ["Москва", "Санкт-Петербург"]


def test_coordinates_are_in_russia() -> None:
    for city, info in CITIES.items():
        assert 41 < info.lat < 70 and 19 < info.lon < 180, city


def test_canonical_place() -> None:
    assert canonical_place("Казань") == Place(city="Казань", region="Республика Татарстан")
    assert canonical_place("Москва") == Place(city="Москва", region="Москва")
    for unknown in ("Kazan", "Houston", "казань"):
        with pytest.raises(KeyError, match="unknown city"):
            canonical_place(unknown)


def test_data_spec_city_table_matches_cities() -> None:
    doc = (REPO_ROOT / "docs" / "DATA_SPEC.md").read_text(encoding="utf-8")
    cell = r" ([^|]+?) \|"
    row = (
        r"^\| ([А-ЯЁ][^|]*?) \|"
        + cell
        + r" ([A-Z][^|]*?) \|"
        + cell
        + " из"
        + cell
        + " в"
        + cell
        + "$"
    )
    rows = re.findall(row, doc, re.M)
    assert {row[0]: row[1:] for row in rows} == {
        city: (info.region, info.name_en, info.region_en, info.genitive, info.accusative)
        for city, info in CITIES.items()
    }
