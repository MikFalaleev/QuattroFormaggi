from __future__ import annotations

from datetime import date

import pytest

from qf.contracts import Place, Quantity
from qf.domain import MAX_PLAUSIBLE_WEIGHT_KG, check_card_values
from tests.factories import make_card, make_card_v2

REQUEST = date(2024, 3, 10)


def test_clean_card_has_no_warnings() -> None:
    assert check_card_values(make_card_v2(), REQUEST) == []
    assert (
        check_card_values(make_card(pickup_date=date(2024, 3, 11), delivery_date=None), REQUEST)
        == []
    )


def test_unknown_city_and_wrong_region() -> None:
    card = make_card_v2(origin=Place(city="Атлантида", region="Море"),
                        destination=Place(city="Казань", region="Пермский край"))  # fmt: skip
    assert check_card_values(card, REQUEST) == ["unknown_city:origin", "unknown_city:destination"]


def test_missing_places_and_dates_are_not_warnings() -> None:
    card = make_card_v2(origin=None, destination=None, pickup_date=None, delivery_date=None)
    assert check_card_values(card, REQUEST) == []


def test_pickup_before_request_and_delivery_before_pickup() -> None:
    card = make_card_v2(pickup_date=date(2024, 3, 9), delivery_date=date(2024, 3, 8))
    assert check_card_values(card, REQUEST) == ["pickup_before_request", "delivery_before_pickup"]


def test_same_day_dates_are_fine() -> None:
    card = make_card_v2(pickup_date=REQUEST, delivery_date=REQUEST)
    assert check_card_values(card, REQUEST) == []


@pytest.mark.parametrize(
    ("weight", "warned"),
    [(Quantity(value=25, unit="t"), False), (Quantity(value=25.1, unit="t"), True),
     (Quantity(value=30000, unit="kg"), True)],
)  # fmt: skip
def test_weight_limit(weight: Quantity, warned: bool) -> None:
    card = make_card_v2(weight_total=weight)
    assert (check_card_values(card, REQUEST) == ["weight_implausible"]) is warned
    assert MAX_PLAUSIBLE_WEIGHT_KG == 25_000.0


def test_per_piece_weight_uses_assumed_single_piece() -> None:
    card = make_card_v2(weight_total=None, weight_per_piece=Quantity(value=30, unit="t"),
                        pieces=None)  # fmt: skip
    assert check_card_values(card, REQUEST) == ["weight_implausible"]
