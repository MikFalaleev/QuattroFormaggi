"""Mass units: exact conversion to kg and rounding rules for rendered text (plan B.3, D-004).

The model copies a mass as written (`{value, unit}`); every conversion is done here.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from qf.contracts import Quantity, WeightUnit

__all__ = [
    "LB_TO_KG",
    "T_TO_KG",
    "format_kg",
    "from_kg",
    "normalize_number",
    "render_value_from_lbs",
    "to_kg",
]

LB_TO_KG: Final = 0.45359237  # exact by definition of the international pound
T_TO_KG: Final = 1000.0
_KG_PER_UNIT: Final[dict[WeightUnit, float]] = {"kg": 1.0, "t": T_TO_KG, "lb": LB_TO_KG}
_LB_TO_KG_EXACT: Final = Decimal("0.45359237")


def to_kg(quantity: Quantity) -> float:
    """Mass in kg, not rounded."""
    return quantity.value * _KG_PER_UNIT[quantity.unit]


def format_kg(kg: float) -> float:
    """Rounding rule for showing a computed mass: 0.1 kg."""
    return round(kg, 1)


def from_kg(kg: float, unit: WeightUnit) -> float:
    """Inverse of `to_kg`, not rounded."""
    return kg / _KG_PER_UNIT[unit]


def normalize_number(x: int | float) -> int | float:
    """Integral floats become ints (13.0 -> 13) so one value has one JSON spelling."""
    if isinstance(x, bool):
        raise TypeError("a bool is not a number here")
    if isinstance(x, int):
        return x
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {x}")
    return int(x) if x.is_integer() else x


def render_value_from_lbs(lbs: int, unit: WeightUnit) -> int | float:
    """The number written in the request text for a source mass of `lbs` pounds.

    lb: the source integer; kg: rounded to an integer; t: rounded to 0.1 t. Rounding is
    half-up on the exact decimal product, so the result never depends on float error.
    The gold answer takes this rendered value, not the source pounds.
    """
    if isinstance(lbs, bool) or not isinstance(lbs, int) or lbs <= 0:
        raise ValueError(f"lbs must be a positive integer, got {lbs!r}")
    if unit == "lb":
        return lbs
    kg = Decimal(lbs) * _LB_TO_KG_EXACT
    if unit == "kg":
        value = kg.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    else:
        value = (kg / 1000).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if value <= 0:
        raise ValueError(f"{lbs} lb rounds to zero in '{unit}'")
    return normalize_number(float(value))
