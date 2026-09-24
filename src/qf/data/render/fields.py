"""Rendering of single card fields into text fragments with evidence (plan 6.4).

Every function takes a `random.Random` and returns text together with the value that text
denotes. The gold answer is built only from these values (invariant E1), so a rounded «12,6 т»
in the text gives the gold `{"value": 12.6, "unit": "t"}`, not the source pounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from random import Random
from typing import Any

from qf.contracts import (
    AnyFieldName,
    AnyLoadFacts,
    FieldName,
    Language,
    Place,
    Quantity,
    RenderedField,
    RequestDraft,
    WeightUnit,
)
from qf.data.render.vocabulary import (
    EN_MACHINERY_PIECE_NOUNS,
    EN_MONTHS,
    EN_MONTHS_SHORT,
    EN_PIECE_NOUNS,
    EN_TONNE_WORDS,
    RELATIVE_DAYS,
    RU_MACHINERY_PIECE_NOUNS,
    RU_MACHINERY_PIECE_SLANG,
    RU_MONTHS_GENITIVE,
    RU_PIECE_NOUNS,
    RU_PIECE_SLANG,
    RU_TONNE,
    VOCABULARIES,
    CardVocabulary,
    ru_plural,
)
from qf.domain import CITIES, render_value_from_lbs, render_value_per_piece

__all__ = [
    "Fragment",
    "build_fragments",
    "format_number",
    "render_date",
    "render_pieces",
    "render_weight",
    "weight_text",
]


@dataclass(frozen=True)
class Fragment:
    """A piece of request text. `fields` are the evidence inside it (each `text` is a
    substring of `text`); `covers` are the coverage keys a layout must place exactly once."""

    text: str
    fields: tuple[RenderedField, ...] = ()
    covers: frozenset[str] = field(default_factory=frozenset)


def _field(text: str, name: AnyFieldName, gold: Any) -> RenderedField:
    return RenderedField(text=text, field=name, gold_value=gold)


def _mention(text: str, evidence: str, name: FieldName, gold: Any) -> Fragment:
    return Fragment(text, (_field(evidence, name, gold),), frozenset({name}))


# --- numbers and quantities ----------------------------------------------------------------


def format_number(value: int | float, lang: Language, rng: Random) -> str:
    """RU: decimal comma and a space (plain or non-breaking) between thousands; EN: decimal
    point and a comma between thousands. Thousands are grouped in about half the cases."""
    if isinstance(value, float):
        text = f"{value:.1f}"
        return text.replace(".", ",") if lang == "ru" else text
    digits = str(value)
    if value < 1000 or rng.random() < 0.5:
        return digits
    groups: list[str] = []
    while digits:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    separator = rng.choice((" ", "\u00a0")) if lang == "ru" else ","
    return separator.join(groups)


def weight_text(
    value: int | float, unit: WeightUnit, lang: Language, rng: Random, *, compact: bool = False
) -> str:
    """«12,6 т», «12 592 кг», «12.6 tonnes»; compact (slang) glues abbreviations: «12,6т»."""
    number = format_number(value, lang, rng)
    one = value == 1
    if lang == "ru":
        words = {
            "kg": ("кг",),
            "t": ("т", ru_plural(value, RU_TONNE) if isinstance(value, int) else RU_TONNE[1]),
            "lb": ("фунт" if one else "фунтов",),
        }[unit]
    else:
        tonnes = ("t", "tonne", "metric ton") if one else EN_TONNE_WORDS
        words = {"kg": ("kg",), "t": tonnes, "lb": ("lb", "lbs")}[unit]
    word = rng.choice(words)
    glued = compact and word in ("т", "кг", "t", "kg")
    return f"{number}{'' if glued else ' '}{word}"


def render_weight(
    lbs: int, lang: Language, unit: WeightUnit, rng: Random, *, compact: bool = False
) -> tuple[str, Quantity]:
    value = render_value_from_lbs(lbs, unit)
    return weight_text(value, unit, lang, rng, compact=compact), Quantity(value=value, unit=unit)


def render_pieces(
    n: int, lang: Language, rng: Random, *, compact: bool = False, machinery: bool = False
) -> str:
    """«22 паллеты», «1 место», «18 палл.» (compact), «22 pallets»; machinery (card_v2) counts
    units: «2 единицы техники», «1 ед.», «2 units»."""
    if lang == "en":
        singular, plural = rng.choice(EN_MACHINERY_PIECE_NOUNS if machinery else EN_PIECE_NOUNS)
        return f"{n} {singular if n == 1 else plural}"
    slang, nouns = ((RU_MACHINERY_PIECE_SLANG, RU_MACHINERY_PIECE_NOUNS) if machinery
                    else (RU_PIECE_SLANG, RU_PIECE_NOUNS))  # fmt: skip
    if compact and rng.random() < 0.6:
        return f"{n} {rng.choice(slang)}"
    return f"{n} {ru_plural(n, rng.choice(nouns))}"


def _machinery(facts: AnyLoadFacts) -> bool:
    return facts.cargo_category == "machinery"  # card_v2 only; never true for card_v1 facts


def _per_piece(facts: AnyLoadFacts, draft: RequestDraft, rng: Random, compact: bool) -> Fragment:
    unit = draft.weight_unit
    value = render_value_per_piece(facts.weight_lbs, facts.pieces, unit)
    pieces = render_pieces(facts.pieces, draft.language, rng, compact=compact,
                           machinery=_machinery(facts))  # fmt: skip
    weight = weight_text(value, unit, draft.language, rng, compact=compact)
    if draft.language == "ru":
        text = f"{pieces} по {weight}"
    else:
        text = rng.choice((f"{pieces}, {weight} each", f"{pieces} at {weight} each"))
    gold = Quantity(value=value, unit=unit)
    fields = (_field(pieces, "pieces", facts.pieces), _field(weight, "weight_per_piece", gold))
    return Fragment(text, fields, frozenset({"pieces", "weight_per_piece"}))


# --- dates ---------------------------------------------------------------------------------


DATE_PATTERNS = 12  # a multiple of the number of written formats in each language


def render_date(
    d: date,
    request_date: date,
    lang: Language,
    style: str,
    rng: Random,
    *,
    preposition: bool,
    pattern: int | None = None,
) -> tuple[str, str]:
    """(text, evidence). Relative words only when `d` is 0-2 days after the request date;
    otherwise a relative style falls back to a written date. EN numeric dates are M/D/Y.
    `pattern` fixes the written format, so the dates of one request look alike."""
    offset = (d - request_date).days
    if style == "relative" and 0 <= offset <= 2:
        word = RELATIVE_DAYS[lang][offset]
        return word, word
    choice = pattern if pattern is not None else rng.randrange(DATE_PATTERNS)
    suffix = ""
    if style == "iso":
        evidence = d.isoformat()
    elif lang == "ru":
        month = RU_MONTHS_GENITIVE[d.month - 1]
        written = (f"{d.day} {month} {d.year}", f"{d.day:02d}.{d.month:02d}.{d.year}")
        evidence = written[choice % 3 // 2]  # patterns 0, 1: words (1 with «г.»); 2: digits
        suffix = " г." if choice % 3 == 1 else ""
    else:
        written_en = (
            f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}",
            f"{EN_MONTHS_SHORT[d.month - 1]} {d.day}, {d.year}",
            f"{d.month}/{d.day}/{d.year}",
            f"{d.month:02d}/{d.day:02d}/{d.year}",
        )
        evidence = written_en[choice % 4]
    text = f"on {evidence}" if lang == "en" and preposition else evidence + suffix
    return text, evidence


def _date_fragment(
    name: FieldName, d: date, draft: RequestDraft, rng: Random, preposition: bool, pattern: int
) -> Fragment:
    text, evidence = render_date(
        d, draft.request_date, draft.language, draft.date_style, rng, preposition=preposition,
        pattern=pattern,
    )  # fmt: skip
    return _mention(text, evidence, name, d)


# --- places --------------------------------------------------------------------------------


def _place_fragments(
    name: FieldName, place: Place, draft: RequestDraft, rng: Random
) -> dict[str, Fragment]:
    """Nominative «Казань», «г. Казань», «Казань (Республика Татарстан)» and a directional
    form: «из Казани» / «в Казань», «from Kazan» / «to Kazan». The gold is always the
    canonical place with its region, even when the text has no region (D-048)."""
    info = CITIES[place.city]
    lang, cyrillic = draft.language, draft.city_lang == "ru"
    city = place.city if cyrillic else info.name_en
    region = info.region if cyrillic else info.region_en
    nominative = [f"{city}"]
    if region != city:
        nominative += [f"{city} ({region})", f"{city}, {region}"]
    if lang == "ru" and cyrillic:
        nominative.append(f"г. {city}")
    prefix = {"origin": {"ru": "из", "en": "from"}, "destination": {"ru": "в", "en": "to"}}[name]
    if lang == "ru" and cyrillic:
        declined = info.genitive if name == "origin" else info.accusative
        directional = rng.choice(
            ((f"{prefix[lang]} {declined}", declined), (f"{prefix[lang]} г. {city}", city))
        )
    else:
        directional = (f"{prefix[lang]} {city}", city)
    suffix = "_from" if name == "origin" else "_to"
    return {
        name: _mention(rng.choice(nominative), city, name, place),
        name + suffix: _mention(directional[0], directional[1], name, place),
    }


# --- notes: conflicts and distractors -------------------------------------------------------


def _weight_conflict(
    facts: AnyLoadFacts, draft: RequestDraft, rng: Random, compact: bool
) -> Fragment:
    value = render_value_from_lbs(draft.conflicts["weight_total"], draft.weight_unit)
    text = weight_text(value, draft.weight_unit, draft.language, rng, compact=compact)
    frames = {
        "ru": ("В накладной при этом указано {v}.", "По данным склада вес {v}.",
               "По документам отправителя вес {v}."),
        "en": ("The waybill, however, says {v}.", "Warehouse data shows {v}.",
               "The shipper's documents show {v}."),
    }[draft.language]  # fmt: skip
    gold = Quantity(value=value, unit=draft.weight_unit)
    return Fragment(
        rng.choice(frames).format(v=text),
        (_field(text, "weight_total", gold),),
        frozenset({"conflict:weight_total"}),
    )


def _pieces_conflict(
    facts: AnyLoadFacts, draft: RequestDraft, rng: Random, compact: bool
) -> Fragment:
    count = draft.conflicts["pieces"]
    text = render_pieces(count, draft.language, rng, compact=compact, machinery=_machinery(facts))
    frames = {
        "ru": ("В накладной при этом {v}.", "По данным склада — {v}.",
               "В спецификации указано {v}."),
        "en": ("The waybill, however, lists {v}.", "Warehouse data shows {v}.",
               "The packing list says {v}."),
    }[draft.language]  # fmt: skip
    return Fragment(
        rng.choice(frames).format(v=text),
        (_field(text, "pieces", count),),
        frozenset({"conflict:pieces"}),
    )


def _distractors(draft: RequestDraft, rng: Random) -> Fragment:
    numbers = draft.distractors
    lang = draft.language
    rate = format_number(numbers["rate_rub"], lang, rng)
    if lang == "ru":
        parts = [f"ставка {rate} ₽", f"номер заявки {numbers['request_no']}",
                 f"рампа № {numbers['dock']}"]  # fmt: skip
        opener = rng.choice(("Для справки: ", "Доп. информация: ", "Также: "))
    else:
        parts = [f"rate RUB {rate}", f"request no. {numbers['request_no']}",
                 f"loading dock {numbers['dock']}"]  # fmt: skip
        opener = rng.choice(("For reference: ", "Also: ", "Additional info: "))
    rng.shuffle(parts)
    return Fragment(opener + ", ".join(parts) + ".", (), frozenset({"distractors"}))


# --- all fragments of one draft ------------------------------------------------------------


def build_fragments(
    facts: AnyLoadFacts,
    draft: RequestDraft,
    rng: Random,
    *,
    compact: bool,
    preposition: bool,
    vocabulary: CardVocabulary = VOCABULARIES["card_v1"],
) -> dict[str, Fragment]:
    """Placeholder name -> fragment for every field the draft puts into the text.

    Placeholders: shipper, category, equipment, pieces, weight, per_piece, origin,
    origin_from, destination, destination_to, pickup, delivery, weight_conflict,
    pieces_conflict, distractors. Dropped fields have no placeholder.
    """
    lang, dropped = draft.language, set(draft.dropped_fields)
    fragments: dict[str, Fragment] = {}
    if "shipper_name" not in dropped:
        fragments["shipper"] = _mention(
            facts.shipper_name, facts.shipper_name, "shipper_name", facts.shipper_name
        )
    if "cargo_category" not in dropped:
        word = rng.choice(vocabulary.category[lang][facts.cargo_category])
        fragments["category"] = _mention(word, word, "cargo_category", facts.cargo_category)
    if "equipment_type" not in dropped:
        words = vocabulary.equipment[lang][facts.equipment_type]
        if compact and lang == "ru":
            words = vocabulary.equipment_slang[facts.equipment_type]
        word = rng.choice(words)
        fragments["equipment"] = _mention(word, word, "equipment_type", facts.equipment_type)
    if draft.weight_mode == "per_piece":
        fragments["per_piece"] = _per_piece(facts, draft, rng, compact)
    else:
        if "pieces" not in dropped:
            text = render_pieces(facts.pieces, lang, rng, compact=compact,
                                 machinery=_machinery(facts))  # fmt: skip
            fragments["pieces"] = _mention(text, text, "pieces", facts.pieces)
        if "weight_total" not in dropped:
            text, gold = render_weight(facts.weight_lbs, lang, draft.weight_unit, rng,
                                       compact=compact)  # fmt: skip
            fragments["weight"] = _mention(text, text, "weight_total", gold)
    places: tuple[tuple[FieldName, Place], ...] = (
        ("origin", facts.origin),
        ("destination", facts.destination),
    )
    for name, place in places:
        if name not in dropped:
            fragments.update(_place_fragments(name, place, draft, rng))
    pattern = rng.randrange(DATE_PATTERNS)  # one written date format per request
    if "pickup_date" not in dropped:
        fragments["pickup"] = _date_fragment("pickup_date", facts.pickup_date, draft, rng,
                                             preposition, pattern)  # fmt: skip
    if "delivery_date" not in dropped:
        fragments["delivery"] = _date_fragment("delivery_date", facts.delivery_date, draft, rng,
                                               preposition, pattern)  # fmt: skip
    if "weight_total" in draft.conflicts:
        fragments["weight_conflict"] = _weight_conflict(facts, draft, rng, compact)
    if "pieces" in draft.conflicts:
        fragments["pieces_conflict"] = _pieces_conflict(facts, draft, rng, compact)
    if draft.distractors:
        fragments["distractors"] = _distractors(draft, rng)
    return fragments
