"""Units: exact conversion of masses to kg and of lengths to metres, rounding rules for
rendered text (plan B.3, D-004, D-081).

The model copies a mass or a length as written (`{value, unit}`); every conversion is done here.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from qf.contracts import Length, LengthUnit, Quantity, WeightUnit

__all__ = [
    "CM_TO_M",
    "LB_TO_KG",
    "T_TO_KG",
    "format_kg",
    "from_kg",
    "normalize_number",
    "render_value_from_lbs",
    "render_value_per_piece",
    "to_kg",
    "to_m",
]

LB_TO_KG: Final = 0.45359237  # exact by definition of the international pound
T_TO_KG: Final = 1000.0
_KG_PER_UNIT: Final[dict[WeightUnit, float]] = {"kg": 1.0, "t": T_TO_KG, "lb": LB_TO_KG}
_LB_TO_KG_EXACT: Final = Decimal("0.45359237")
CM_TO_M: Final = 0.01
_M_PER_UNIT: Final[dict[LengthUnit, float]] = {"m": 1.0, "cm": CM_TO_M}


def to_kg(quantity: Quantity) -> float:
    """Mass in kg, not rounded."""
    return quantity.value * _KG_PER_UNIT[quantity.unit]


def to_m(length: Length) -> float:
    """Length in metres, not rounded (card_v2 dimensions)."""
    return length.value * _M_PER_UNIT[length.unit]


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
    return render_value_per_piece(lbs, 1, unit)


def render_value_per_piece(lbs: int, pieces: int, unit: WeightUnit) -> int | float:
    """The number written for the mass of one of `pieces` equal pieces of `lbs` pounds in total.

    Same rounding rules as `render_value_from_lbs` (which is the case `pieces == 1`).
    """
    for name, value in (("lbs", lbs), ("pieces", pieces)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if unit == "lb":
        rounded = (Decimal(lbs) / pieces).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    elif unit == "kg":
        kg = Decimal(lbs) * _LB_TO_KG_EXACT / pieces
        rounded = kg.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    else:
        tonnes = Decimal(lbs) * _LB_TO_KG_EXACT / pieces / 1000
        rounded = tonnes.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if rounded <= 0:
        raise ValueError(f"{lbs} lb / {pieces} rounds to zero in '{unit}'")
    return normalize_number(float(rounded))
