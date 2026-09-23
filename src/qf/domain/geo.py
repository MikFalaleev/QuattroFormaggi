"""The 20 cities of the dataset with English and Russian names (plan B.1, step 4).

Russian names are a manual translation checked by a human in the step 4 report.
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
    state: str  # two-letter US state code
    state_name_en: str
    name_ru: str
    state_name_ru: str


CITIES: Final[Mapping[str, CityInfo]] = MappingProxyType(
    {
        "Atlanta": CityInfo("GA", "Georgia", "Атланта", "Джорджия"),
        "Charlotte": CityInfo("NC", "North Carolina", "Шарлотт", "Северная Каролина"),
        "Chicago": CityInfo("IL", "Illinois", "Чикаго", "Иллинойс"),
        "Columbus": CityInfo("OH", "Ohio", "Колумбус", "Огайо"),
        "Dallas": CityInfo("TX", "Texas", "Даллас", "Техас"),
        "Denver": CityInfo("CO", "Colorado", "Денвер", "Колорадо"),
        "Detroit": CityInfo("MI", "Michigan", "Детройт", "Мичиган"),
        "Houston": CityInfo("TX", "Texas", "Хьюстон", "Техас"),
        "Indianapolis": CityInfo("IN", "Indiana", "Индианаполис", "Индиана"),
        "Kansas City": CityInfo("MO", "Missouri", "Канзас-Сити", "Миссури"),
        "Las Vegas": CityInfo("NV", "Nevada", "Лас-Вегас", "Невада"),
        "Los Angeles": CityInfo("CA", "California", "Лос-Анджелес", "Калифорния"),
        "Memphis": CityInfo("TN", "Tennessee", "Мемфис", "Теннесси"),
        "Miami": CityInfo("FL", "Florida", "Майами", "Флорида"),
        "Minneapolis": CityInfo("MN", "Minnesota", "Миннеаполис", "Миннесота"),
        "New York": CityInfo("NY", "New York", "Нью-Йорк", "Нью-Йорк"),
        "Philadelphia": CityInfo("PA", "Pennsylvania", "Филадельфия", "Пенсильвания"),
        "Phoenix": CityInfo("AZ", "Arizona", "Финикс", "Аризона"),
        "Portland": CityInfo("OR", "Oregon", "Портленд", "Орегон"),
        "Seattle": CityInfo("WA", "Washington", "Сиэтл", "Вашингтон"),
    }
)


def canonical_place(city_en: str, state: str) -> Place:
    """The card value for a known city; KeyError for an unknown city or a wrong state."""
    info = CITIES.get(city_en)
    if info is None:
        raise KeyError(f"unknown city {city_en!r}")
    if info.state != state:
        raise KeyError(f"{city_en} is in {info.state}, not {state}")
    return Place(city=city_en, state=state)
