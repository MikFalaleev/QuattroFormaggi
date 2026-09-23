from __future__ import annotations

import re
from datetime import date
from random import Random

import pytest

from qf.contracts import Quantity, RequestDraft
from qf.data import HARD_CASES
from qf.data.render import (
    Fragment,
    HardCaseNotApplicable,
    RenderError,
    build_fragments,
    fill_layout,
    format_number,
    render_date,
    ru_plural,
)
from qf.data.render.fields import render_pieces, render_weight, weight_text
from qf.data.render.vocabulary import RU_PIECE_NOUNS, RU_TONNE
from qf.domain import render_value_from_lbs
from tests.generation import facts_of


def parse_number(text: str) -> float:
    """The number at the start of a rendered quantity: «12 592 кг», «12,6 т», «12,592 kg»."""
    match = re.match(r"[\d\s ,.]+", text)
    assert match, text
    raw = match.group().strip()
    if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):  # EN thousands
        raw = raw.replace(",", "")
    return float(raw.replace(" ", "").replace(" ", "").replace(",", "."))


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("unit", ["kg", "t"])
def test_weight_formats_parse_back(lang: str, unit: str) -> None:
    for i, lbs in enumerate(range(10_000, 45_001, 350)):
        text, gold = render_weight(lbs, lang, unit, Random(i))  # type: ignore[arg-type]
        assert gold == Quantity(value=render_value_from_lbs(lbs, unit), unit=unit)  # type: ignore[arg-type]
        assert parse_number(text) == gold.value, text


def test_ru_decimal_comma_and_en_point() -> None:
    assert format_number(12.6, "ru", Random(0)) == "12,6"
    assert format_number(12.6, "en", Random(0)) == "12.6"
    grouped = {format_number(12592, "ru", Random(i)) for i in range(40)}
    assert grouped == {"12592", "12 592", "12 592"}
    assert {format_number(12592, "en", Random(i)) for i in range(40)} == {"12592", "12,592"}
    assert format_number(572, "ru", Random(0)) == "572"


def test_ru_plural() -> None:
    forms = ("паллета", "паллеты", "паллет")
    expected = {1: "паллета", 2: "паллеты", 4: "паллеты", 5: "паллет", 11: "паллет",
                12: "паллет", 14: "паллет", 21: "паллета", 22: "паллеты", 25: "паллет",
                111: "паллет", 101: "паллета"}  # fmt: skip
    assert {n: ru_plural(n, forms) for n in expected} == expected


def test_tonnes_agree_with_the_number() -> None:
    def words(value: float) -> set[str]:
        return {weight_text(value, "t", "ru", Random(i)).split(" ", 1)[1] for i in range(30)}

    assert words(21) == {"т", "тонна"}
    assert words(13) == {"т", "тонн"}
    assert words(22) == {"т", "тонны"}
    assert words(12.6) == {"т", "тонны"}
    assert words(0.6) == {"т", "тонны"}


def test_pieces_agree_with_the_number() -> None:
    for n in (1, 2, 5, 11, 21, 22, 28):
        text = render_pieces(n, "ru", Random(n))
        number, noun = text.split(" ", 1)
        assert int(number) == n
        assert noun in {ru_plural(n, forms) for forms in RU_PIECE_NOUNS}
    assert render_pieces(1, "en", Random(0)).split()[1] in {"pallet", "piece", "package"}
    assert RU_TONNE == ("тонна", "тонны", "тонн")


def test_relative_date_only_within_2_days() -> None:
    request = date(2022, 3, 1)
    words = ["сегодня", "завтра", "послезавтра"]
    for offset, word in enumerate(words):
        day = date(2022, 3, 1 + offset)
        assert render_date(day, request, "ru", "relative", Random(0), preposition=True) == (
            word, word,
        )  # fmt: skip
    text, evidence = render_date(date(2022, 3, 4), request, "ru", "relative", Random(0),
                                 preposition=True)  # fmt: skip
    assert evidence not in words and "2022" in evidence
    text, _ = render_date(date(2022, 2, 28), request, "en", "relative", Random(0), preposition=True)
    assert text.startswith("on ") and "2022" in text


def test_relative_date_gold_correct_across_month_boundary() -> None:
    facts = facts_of()[0]
    draft = _draft(request_date=date(2023, 1, 31), date_style="relative")
    shifted = facts.model_copy(update={"pickup_date": date(2023, 2, 1),
                                       "delivery_date": date(2023, 2, 2)})  # fmt: skip
    fragments = build_fragments(shifted, draft, Random(0), compact=False, preposition=True)
    assert fragments["pickup"].text == "завтра"
    assert fragments["pickup"].fields[0].gold_value == date(2023, 2, 1)
    assert fragments["delivery"].text == "послезавтра"


def test_en_numeric_date_is_month_first() -> None:
    texts = {render_date(date(2022, 3, 5), date(2022, 3, 1), "en", "text", Random(i),
                         preposition=False)[1] for i in range(40)}  # fmt: skip
    assert texts == {"March 5, 2022", "Mar 5, 2022", "3/5/2022", "03/05/2022"}
    ru = {render_date(date(2022, 3, 5), date(2022, 3, 1), "ru", "text", Random(i),
                      preposition=False)[1] for i in range(40)}  # fmt: skip
    assert ru == {"5 марта 2022", "05.03.2022"}


def _draft(**changes: object) -> RequestDraft:
    base = {
        "language": "ru", "request_date": date(2021, 12, 30), "weight_unit": "kg",
        "weight_mode": "total", "date_style": "iso", "city_lang": "ru", "dropped_fields": [],
        "conflicts": {}, "distractors": {}, "hard_cases": [], "ood_reason": None,
    }  # fmt: skip
    return RequestDraft.model_validate({**base, **changes})


def test_places_are_canonical_whatever_the_spelling() -> None:
    facts = facts_of()[0]  # Воронеж -> Казань
    seen = set()
    for city_lang in ("ru", "en"):
        for i in range(30):
            fragments = build_fragments(facts, _draft(city_lang=city_lang), Random(i),
                                        compact=False, preposition=True)  # fmt: skip
            for key in ("origin", "origin_from"):
                field = fragments[key].fields[0]
                assert field.gold_value == facts.origin
                assert field.text in fragments[key].text
                seen.add(fragments[key].text)
    assert {"из Воронежа", "из г. Воронеж", "Voronezh", "из Voronezh"} <= seen


def test_per_piece_fragment_carries_both_fields() -> None:
    facts = facts_of()[2]  # 27761 lb, 22 pieces
    fragments = build_fragments(facts, _draft(weight_mode="per_piece"), Random(1), compact=False,
                                preposition=True)  # fmt: skip
    piece = fragments["per_piece"]
    assert {f.field for f in piece.fields} == {"pieces", "weight_per_piece"}
    assert " по 572 кг" in piece.text
    assert "pieces" not in fragments and "weight" not in fragments


def test_layout_that_loses_a_field_is_an_error() -> None:
    fragments = {"shipper": Fragment("ACME", (), frozenset({"shipper_name"}))}
    with pytest.raises(RenderError, match="did not place"):
        fill_layout((("Добрый день!",),), fragments, Random(0), " ")
    text, _ = fill_layout((("Добрый день!",), ("От {shipper}.",)), fragments, Random(0), " ")
    assert text == "Добрый день! От ACME."


# --- hard cases ----------------------------------------------------------------------------


def _apply(name: str, facts_index: int = 0, seed: int = 0, **draft: object) -> RequestDraft:
    return HARD_CASES.get(name)().apply(_draft(**draft), facts_of()[facts_index], Random(seed))


def test_dropped_fields_drops_one_to_three() -> None:
    sizes = {len(_apply("dropped_fields", seed=s).dropped_fields) for s in range(60)}
    assert sizes == {1, 2, 3}


def test_conflicts_differ_after_rounding() -> None:
    for seed in range(50):
        draft = _apply("conflict_weight", facts_index=2, seed=seed, weight_unit="t")
        other = draft.conflicts["weight_total"]
        assert render_value_from_lbs(other, "t") != render_value_from_lbs(27761, "t")
        pieces = _apply("conflict_pieces", facts_index=0, seed=seed).conflicts["pieces"]
        assert pieces in (2, 3, 4)  # load 1 has a single piece: shifts go upwards


class _SameRandom(Random):
    """Always 1.0 for uniform(): the second weight equals the first."""

    def uniform(self, a: float, b: float) -> float:
        return 1.0


def test_conflict_weight_gives_up_after_ten_attempts() -> None:
    case = HARD_CASES.get("conflict_weight")()
    with pytest.raises(HardCaseNotApplicable, match="no distinct second weight"):
        case.apply(_draft(), facts_of()[0], _SameRandom(0))


class _CollidingRandom(Random):
    """randrange() returns 7 first, then falls back to the real generator."""

    def __init__(self) -> None:
        super().__init__(0)
        self.first = True

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:  # type: ignore[override]
        if self.first and start <= 7:
            self.first = False
            return 7
        return super().randrange(start, stop, step)


def test_distractors_never_equal_card_numbers() -> None:
    facts = facts_of()[6]  # LOAD00000007: 9 pieces
    seven = facts.model_copy(update={"pieces": 7})
    draft = HARD_CASES.get("distractor_numbers")().apply(_draft(), seven, _CollidingRandom())
    assert 7 not in draft.distractors.values()
    assert set(draft.distractors) == {"rate_rub", "request_no", "dock"}


def test_relative_date_and_city_switch() -> None:
    facts = facts_of()[0]
    draft = _apply("relative_date", seed=3)
    assert draft.date_style == "relative"
    assert 0 <= (facts.pickup_date - draft.request_date).days <= 2
    assert _apply("city_lang_switch").city_lang == "en"
    assert _apply("city_lang_switch", language="en", city_lang="en").city_lang == "ru"


def test_hard_case_output_is_validated() -> None:
    draft = _apply("per_piece_weight")
    assert draft.weight_mode == "per_piece" and draft.hard_cases == ["per_piece_weight"]
    light = facts_of()[0].model_copy(update={"weight_lbs": 100, "pieces": 28})
    assert (
        HARD_CASES.get("per_piece_weight")()
        .apply(_draft(weight_unit="t"), light, Random(0))
        .weight_unit
        == "kg"
    )  # 1.6 kg per piece rounds to 0 t


def test_sentences_start_with_capital_letters_except_slang() -> None:
    fragments = {"origin_from": Fragment("из Казани", (), frozenset({"origin"}))}
    layout = (("Привет!",), ("{origin_from}, нужна машина.",))
    text, _ = fill_layout(layout, fragments, Random(0), " ")
    assert text == "Привет! Из Казани, нужна машина."
    slang, _ = fill_layout(layout, fragments, Random(0), "\n", capitalize=False)
    assert slang == "Привет!\nиз Казани, нужна машина."


def test_line_break_ends_a_sentence_without_joiner() -> None:
    text, _ = fill_layout((("Dear team,\n",), ("Please quote.",)), {}, Random(0), " ")
    assert text == "Dear team,\nPlease quote."


def test_dates_of_one_request_share_a_format() -> None:
    facts = facts_of()[1]  # pickup and delivery on different days
    for seed in range(40):
        for lang in ("ru", "en"):
            fragments = build_fragments(facts, _draft(language=lang, date_style="text"),
                                        Random(seed), compact=False, preposition=False)  # fmt: skip
            pickup, delivery = (fragments[k].fields[0].text for k in ("pickup", "delivery"))
            assert re.sub(r"\d", "0", pickup) == re.sub(r"\d", "0", delivery), (pickup, delivery)
            assert fragments["pickup"].text.endswith(" г.") == fragments["delivery"].text.endswith(
                " г."
            )


def test_no_double_full_stop_and_glued_words_only_for_units() -> None:
    fragments = {"pickup": Fragment("23 июня 2022 г.", (), frozenset({"pickup_date"}))}
    text, _ = fill_layout((("Грузимся {pickup}.",),), fragments, Random(0), " ")
    assert text == "Грузимся 23 июня 2022 г."
    compact = {weight_text(5.4, "t", "ru", Random(i), compact=True) for i in range(30)}
    assert compact == {"5,4т", "5,4 тонны"}
