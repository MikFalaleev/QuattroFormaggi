"""Independent checks of the values in a card (step 18): pure functions, no model involved.

Valid JSON does not prove correct values. These checks look at a parsed card (card_v1 or
card_v2) and its request date and return warning codes; they are used by `qf extract` and can
be reused in evaluation reports. A warning is a reason for a person to look, not a verdict.
"""

from __future__ import annotations

from datetime import date
from typing import Final

from qf.contracts import ShipmentCard, ShipmentCardV2
from qf.domain.geo import CITIES
from qf.domain.rules import total_weight_kg

__all__ = [
    "MAX_PLAUSIBLE_WEIGHT_KG",
    "check_card_values",
]

MAX_PLAUSIBLE_WEIGHT_KG: Final = 25_000.0
"""The heaviest total mass that one semi-trailer carries (payload of a 20 t truck with margin);
a larger value is a warning, not an error: the request may be for several vehicles."""


def check_card_values(card: ShipmentCard | ShipmentCardV2, request_date: date) -> list[str]:
    """Warning codes, in a fixed order; an empty list means nothing suspicious.

    - `unknown_city:<origin|destination>` — the city (or its region) is not in the catalog;
    - `pickup_before_request` — `pickup_date` is earlier than the request date;
    - `delivery_before_pickup` — `delivery_date` is earlier than `pickup_date`;
    - `weight_implausible` — the total mass is above `MAX_PLAUSIBLE_WEIGHT_KG`.
    """
    warnings = []
    for name in ("origin", "destination"):
        place = getattr(card, name)
        if place is None:
            continue
        info = CITIES.get(place.city)
        if info is None or info.region != place.region:
            warnings.append(f"unknown_city:{name}")
    if card.pickup_date is not None and card.pickup_date < request_date:
        warnings.append("pickup_before_request")
    if (
        card.pickup_date is not None
        and card.delivery_date is not None
        and card.delivery_date < card.pickup_date
    ):
        warnings.append("delivery_before_pickup")
    kg = total_weight_kg(card)
    if kg is not None and kg > MAX_PLAUSIBLE_WEIGHT_KG:
        warnings.append("weight_implausible")
    return warnings
