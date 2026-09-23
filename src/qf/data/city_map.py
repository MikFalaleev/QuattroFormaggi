"""Source (US) cities -> Russian cities of the card (D-047).

The raw dataset stays as published; the map is applied when LoadFacts are built (step 5), and
its file hash goes into that run's manifest.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from qf.common import QFError, StrictConfig, load_yaml_config
from qf.contracts import Place
from qf.domain import CITIES, canonical_place

__all__ = ["CityMap", "load_city_map"]


class CityMap(StrictConfig):
    """A one-to-one map from source city names onto cities of `qf.domain.CITIES`."""

    version: Literal["city_map_ru_v1"]
    cities: dict[str, str]

    @model_validator(mode="after")
    def _one_to_one_onto_catalog(self) -> CityMap:
        unknown = sorted(set(self.cities.values()) - set(CITIES))
        if unknown:
            raise ValueError(f"cities not in qf.domain.CITIES: {unknown}")
        shared = sorted(city for city, n in Counter(self.cities.values()).items() if n > 1)
        if shared:
            raise ValueError(f"several source cities map to {shared}")
        return self

    def place(self, source_city: str) -> Place:
        """The card value for a city named in the source tables."""
        if source_city not in self.cities:
            raise QFError(f"source city {source_city!r} is not in city map {self.version}")
        return canonical_place(self.cities[source_city])


def load_city_map(path: Path) -> CityMap:
    return load_yaml_config(path, CityMap)
