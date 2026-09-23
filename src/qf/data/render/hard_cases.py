"""Hard cases: controlled difficulties applied to a draft before rendering (plan 6.6).

Each case returns a new, validated draft and records its name in `hard_cases`. A case that
cannot be applied to a load raises `HardCaseNotApplicable`; the generator then picks another.
"""

from __future__ import annotations

import contextlib
from datetime import timedelta
from random import Random
from typing import Any, Final

from qf.contracts import FieldName, LoadFacts, Quantity, RequestDraft
from qf.data.registries import HARD_CASES
from qf.data.render.base import HardCaseNotApplicable
from qf.domain import render_value_from_lbs, render_value_per_piece, to_kg

__all__ = [
    "DROPPABLE_FIELDS",
    "MAX_ATTEMPTS",
    "MIN_WEIGHT_DIFFERENCE_KG",
    "CityLangSwitch",
    "ConflictPieces",
    "ConflictWeight",
    "DistractorNumbers",
    "DroppedFields",
    "PerPieceWeight",
    "RelativeDate",
]

DROPPABLE_FIELDS: Final[tuple[FieldName, ...]] = (
    "shipper_name", "cargo_category", "equipment_type", "pieces", "weight_total",
    "origin", "destination", "pickup_date", "delivery_date",
)  # fmt: skip
MAX_ATTEMPTS: Final = 10
_DISTRACTOR_RANGES: Final = (
    ("rate_rub", 40_000, 400_000, 500),  # a rate in rubles, a multiple of 500
    ("request_no", 10_000, 99_999, 1),
    ("dock", 1, 40, 1),
)
MIN_WEIGHT_DIFFERENCE_KG: Final = 0.5  # the weight tolerance of the eval metrics (step 9)


def _with(draft: RequestDraft, name: str, **changes: Any) -> RequestDraft:
    """A validated copy (model_copy would skip validation) with `name` added to hard_cases."""
    data = {**draft.model_dump(), **changes, "hard_cases": [*draft.hard_cases, name]}
    return RequestDraft.model_validate(data)


class _HardCase:
    name: str

    def applicable(self, facts: LoadFacts) -> bool:
        return True


@HARD_CASES.register("dropped_fields")
class DroppedFields(_HardCase):
    """1-3 fields are left out of the text; they become null and missing_fields follows."""

    name = "dropped_fields"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        chosen = set(rng.sample(DROPPABLE_FIELDS, rng.randint(1, 3)))
        dropped = [name for name in DROPPABLE_FIELDS if name in chosen]
        return _with(draft, self.name, dropped_fields=dropped)


@HARD_CASES.register("per_piece_weight")
class PerPieceWeight(_HardCase):
    """«22 паллеты по 572 кг»: weight_per_piece and pieces, weight_total stays null."""

    name = "per_piece_weight"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        unit = draft.weight_unit
        try:
            render_value_per_piece(facts.weight_lbs, facts.pieces, unit)
        except ValueError:
            unit = "kg"  # a light piece rounds to 0 t; kilograms always work
        return _with(draft, self.name, weight_mode="per_piece", weight_unit=unit)


@HARD_CASES.register("conflict_weight")
class ConflictWeight(_HardCase):
    """A second, different total weight in the text: weight_total becomes a conflict."""

    name = "conflict_weight"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        unit = draft.weight_unit
        first = render_value_from_lbs(facts.weight_lbs, unit)
        for _ in range(MAX_ATTEMPTS):
            other_lbs = round(facts.weight_lbs * rng.uniform(0.8, 1.2))
            if other_lbs <= 0:
                continue
            second = render_value_from_lbs(other_lbs, unit)
            difference = abs(to_kg(Quantity(value=first, unit=unit))
                             - to_kg(Quantity(value=second, unit=unit)))  # fmt: skip
            if second != first and difference > MIN_WEIGHT_DIFFERENCE_KG:
                return _with(draft, self.name, conflicts={"weight_total": other_lbs})
        raise HardCaseNotApplicable(f"{facts.load_id}: no distinct second weight")


@HARD_CASES.register("conflict_pieces")
class ConflictPieces(_HardCase):
    """A second number of pieces, pieces ± 1-3 and at least 1: pieces becomes a conflict."""

    name = "conflict_pieces"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        shift = rng.randint(1, 3)
        other = facts.pieces + rng.choice((-1, 1)) * shift
        if other < 1:  # a downward shift below one piece becomes an upward one
            other = facts.pieces + shift
        return _with(draft, self.name, conflicts={"pieces": other})


@HARD_CASES.register("relative_date")
class RelativeDate(_HardCase):
    """The request date is 0-2 days before pickup and dates are written as «завтра» etc."""

    name = "relative_date"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        request_date = facts.pickup_date - timedelta(days=rng.randint(0, 2))
        return _with(draft, self.name, request_date=request_date, date_style="relative")


@HARD_CASES.register("distractor_numbers")
class DistractorNumbers(_HardCase):
    """Numbers that are not card fields: a rate in rubles, a request number, a dock number.
    None of them may equal a number of the card (pieces, rendered weights)."""

    name = "distractor_numbers"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        taken = {facts.pieces, *_weight_numbers(facts, draft)}
        numbers: dict[str, int] = {}
        for key, low, high, step in _DISTRACTOR_RANGES:
            for _ in range(MAX_ATTEMPTS):
                value = rng.randrange(low, high + 1, step)
                if value not in taken:
                    numbers[key] = value
                    taken.add(value)
                    break
            else:
                raise HardCaseNotApplicable(f"{facts.load_id}: distractor {key} collides")
        return _with(draft, self.name, distractors=numbers)


def _weight_numbers(facts: LoadFacts, draft: RequestDraft) -> set[int | float]:
    values: set[int | float] = {render_value_from_lbs(facts.weight_lbs, draft.weight_unit)}
    with contextlib.suppress(ValueError):  # a piece may round to 0 t
        values.add(render_value_per_piece(facts.weight_lbs, facts.pieces, draft.weight_unit))
    return values


@HARD_CASES.register("city_lang_switch")
class CityLangSwitch(_HardCase):
    """Cities in the other language: «Kazan» in a Russian request, «Казань» in an English one.
    The gold is still the canonical Russian city with its region."""

    name = "city_lang_switch"

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: Random) -> RequestDraft:
        return _with(draft, self.name, city_lang="en" if draft.language == "ru" else "ru")
