from __future__ import annotations

import random

import pytest

from qf.contracts import Quantity
from qf.domain import (
    LB_TO_KG,
    format_kg,
    from_kg,
    normalize_number,
    render_value_from_lbs,
    render_value_per_piece,
    to_kg,
)


def test_lb_factor_exact() -> None:
    assert LB_TO_KG == 0.45359237
    assert to_kg(Quantity(value=1, unit="lb")) == 0.45359237
    assert to_kg(Quantity(value=27761, unit="lb")) == pytest.approx(12592.18778357)


def test_roundtrip_kg_lb() -> None:
    rng = random.Random(4)
    for _ in range(1000):
        quantity = Quantity(value=rng.randint(10_000, 45_000), unit="lb")
        assert abs(from_kg(to_kg(quantity), "lb") - quantity.value) < 1e-6


def test_tonnes() -> None:
    assert to_kg(Quantity(value=12.6, unit="t")) == pytest.approx(12600.0)
    assert to_kg(Quantity(value=13, unit="t")) == 13000.0
    assert to_kg(Quantity(value=12592, unit="kg")) == 12592.0
    assert from_kg(12600.0, "t") == pytest.approx(12.6)
    assert from_kg(500.0, "kg") == 500.0


def test_format_kg_rounds_to_tenth() -> None:
    assert format_kg(12592.18778357) == 12592.2
    assert format_kg(to_kg(Quantity(value=1, unit="lb"))) == 0.5


def test_render_rounding_rules() -> None:
    assert render_value_from_lbs(27761, "lb") == 27761
    assert render_value_from_lbs(27761, "kg") == 12592  # 12592.19 kg
    assert render_value_from_lbs(27761, "t") == 12.6  # 12.592 t
    assert isinstance(render_value_from_lbs(27761, "kg"), int)
    tonnes = render_value_from_lbs(28660, "t")  # 12999.96 kg -> 13.0 t -> 13
    assert tonnes == 13
    assert isinstance(tonnes, int)


def test_render_rounds_half_up_on_the_exact_product() -> None:
    # 10000 lb = 4535.9237 kg exactly; 4.5359237 t rounds to 4.5, 4535.9237 kg to 4536.
    assert render_value_from_lbs(10000, "t") == 4.5
    assert render_value_from_lbs(10000, "kg") == 4536
    # 45000 lb = 20411.65665 kg -> 20.4 t
    assert render_value_from_lbs(45000, "t") == 20.4


def test_render_rejects_bad_input() -> None:
    for bad in (0, -5, True, 12.5):
        with pytest.raises(ValueError, match="positive integer"):
            render_value_from_lbs(bad, "kg")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rounds to zero"):
        render_value_from_lbs(100, "t")


def test_normalize_number() -> None:
    assert normalize_number(13.0) == 13
    assert isinstance(normalize_number(13.0), int)
    assert normalize_number(12.6) == 12.6
    assert normalize_number(7) == 7
    with pytest.raises(ValueError, match="finite"):
        normalize_number(float("inf"))
    with pytest.raises(TypeError):
        normalize_number(True)


def test_render_value_per_piece() -> None:
    assert render_value_per_piece(27761, 22, "kg") == 572  # 12592.19 / 22 = 572.37
    assert render_value_per_piece(27761, 22, "t") == 0.6
    assert render_value_per_piece(27761, 22, "lb") == 1262  # 1261.86
    assert render_value_per_piece(27761, 1, "t") == render_value_from_lbs(27761, "t")
    with pytest.raises(ValueError, match="pieces must be a positive integer"):
        render_value_per_piece(27761, 0, "kg")
    with pytest.raises(ValueError, match="rounds to zero"):
        render_value_per_piece(100, 28, "t")
