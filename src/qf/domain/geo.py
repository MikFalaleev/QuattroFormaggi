"""Russian cities of the card: canonical names, regions, English names, cases (D-047, D-048).

Canonical city and region are Russian names; the region is the official name of the federal
subject (Constitution of the Russian Federation, art. 65); for a federal city it equals the city.
Names, English names (English Wikipedia titles) and coordinates were taken from ru.wikipedia and
city-in-region membership was checked against Wikidata (P131) on 2026-09-23. Grammatical cases
are written by hand. Which source (US) city becomes which of these is data, not domain logic:
`configs/data/city_map_ru_v1.yaml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from qf.contracts import Place

__all__ = ["CITIES", "CityInfo", "canonical_place"]


@dataclass(frozen=True)
class CityInfo:
    region: str  # canonical region, part of the card's Place
    name_en: str
    region_en: str
    genitive: str  # «из Казани»
    accusative: str  # «в Казань»
    lat: float
    lon: float


CITIES: Final[Mapping[str, CityInfo]] = MappingProxyType(
    {
        "Москва": CityInfo(
            "Москва", "Moscow", "Moscow", "Москвы", "Москву", 55.7506, 37.6175
        ),
        "Санкт-Петербург": CityInfo(
            "Санкт-Петербург", "Saint Petersburg", "Saint Petersburg",
            "Санкт-Петербурга", "Санкт-Петербург", 59.95, 30.3167,
        ),
        "Тула": CityInfo("Тульская область", "Tula", "Tula Oblast", "Тулы", "Тулу", 54.2, 37.6167),
        "Воронеж": CityInfo(
            "Воронежская область", "Voronezh", "Voronezh Oblast", "Воронежа", "Воронеж",
            51.6717, 39.2106,
        ),
        "Ростов-на-Дону": CityInfo(
            "Ростовская область", "Rostov-on-Don", "Rostov Oblast", "Ростова-на-Дону",
            "Ростов-на-Дону", 47.2406, 39.7106,
        ),
        "Краснодар": CityInfo(
            "Краснодарский край", "Krasnodar", "Krasnodar Krai", "Краснодара", "Краснодар",
            45.0333, 38.9833,
        ),
        "Волгоград": CityInfo(
            "Волгоградская область", "Volgograd", "Volgograd Oblast", "Волгограда", "Волгоград",
            48.7117, 44.5139,
        ),
        "Нижний Новгород": CityInfo(
            "Нижегородская область", "Nizhny Novgorod", "Nizhny Novgorod Oblast",
            "Нижнего Новгорода", "Нижний Новгород", 56.3269, 44.0075,
        ),
        "Казань": CityInfo(
            "Республика Татарстан", "Kazan", "Tatarstan", "Казани", "Казань", 55.7908, 49.1144
        ),
        "Самара": CityInfo(
            "Самарская область", "Samara", "Samara Oblast", "Самары", "Самару", 53.1833, 50.1167
        ),
        "Уфа": CityInfo(
            "Республика Башкортостан", "Ufa", "Bashkortostan", "Уфы", "Уфу", 54.7333, 55.9667
        ),
        "Пермь": CityInfo(
            "Пермский край", "Perm", "Perm Krai", "Перми", "Пермь", 58.0139, 56.2489
        ),
        "Екатеринбург": CityInfo(
            "Свердловская область", "Yekaterinburg", "Sverdlovsk Oblast", "Екатеринбурга",
            "Екатеринбург", 56.8333, 60.5833,
        ),
        "Челябинск": CityInfo(
            "Челябинская область", "Chelyabinsk", "Chelyabinsk Oblast", "Челябинска",
            "Челябинск", 55.15, 61.4,
        ),
        "Омск": CityInfo(
            "Омская область", "Omsk", "Omsk Oblast", "Омска", "Омск", 54.9667, 73.3833
        ),
        "Новосибирск": CityInfo(
            "Новосибирская область", "Novosibirsk", "Novosibirsk Oblast", "Новосибирска",
            "Новосибирск", 55.0167, 82.9167,
        ),
        "Барнаул": CityInfo(
            "Алтайский край", "Barnaul", "Altai Krai", "Барнаула", "Барнаул", 53.3486, 83.7764
        ),
        "Томск": CityInfo(
            "Томская область", "Tomsk", "Tomsk Oblast", "Томска", "Томск", 56.4886, 84.9522
        ),
        "Кемерово": CityInfo(
            "Кемеровская область — Кузбасс", "Kemerovo", "Kemerovo Oblast", "Кемерово",
            "Кемерово", 55.3542, 86.0897,
        ),
        "Красноярск": CityInfo(
            "Красноярский край", "Krasnoyarsk", "Krasnoyarsk Krai", "Красноярска", "Красноярск",
            56.0121, 92.8713,
        ),
    }
)  # fmt: skip


def canonical_place(city: str) -> Place:
    """The card value for a known city (canonical Russian name); KeyError for an unknown one."""
    info = CITIES.get(city)
    if info is None:
        raise KeyError(f"unknown city {city!r}")
    return Place(city=city, region=info.region)
