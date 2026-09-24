"""Rendering of the special conditions of card_v2 (sub-step V3, D-094).

Each condition of the facts becomes one fragment `cond_<kind>` whose evidence carries the gold
condition as written: the temperature bounds, the list values in the facts' order, the
dimensions in the unit of the draft. A condition named without values keeps its kind and loses
its values; such phrases contain no word of any value list. Only temperature phrases contain
numbers; one-sided bounds are always «не выше» / «не ниже» (never a bare «до»), and the
«Д×Ш×В» form always carries its label, so every text reads unambiguously on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from typing import Final, Literal

from qf.contracts import (
    Dimension,
    Language,
    Length,
    LoadFactsV2,
    OversizeCondition,
    PackagingCondition,
    RenderedField,
    RequestDraft,
    SecuringCondition,
    SensorsCondition,
    SpecialCondition,
    TemperatureCondition,
)
from qf.data.render.fields import Fragment, format_number
from qf.data.render.vocabulary import (
    EN_PACKAGING,
    EN_SECURING,
    EN_SENSORS,
    RU_PACKAGING,
    RU_SECURING,
    RU_SENSORS,
)
from qf.domain import normalize_number

__all__ = ["CONDITION_PREFIX", "ConditionStyle", "condition_fragments", "rendered_length"]

ConditionStyle = Literal["prose", "list", "slang"]
CONDITION_PREFIX: Final = "cond_"
Frames = tuple[str, ...]
_DIMENSIONS: Final[tuple[Dimension, ...]] = ("length", "width", "height")

# {v} is the values text (the evidence); frames never start with it (the layout capitalizes).
_WITH_VALUES: Final[dict[tuple[str, Language, ConditionStyle], Frames]] = {
    ("temperature", "ru", "prose"): ("Температурный режим: {v}.", "Температура перевозки — {v}.",
                                     "Режим {v}."),
    ("temperature", "en", "prose"): ("Temperature: {v}.", "Transport temperature: {v}.",
                                     "Keep temperature {v}."),
    ("temperature", "ru", "list"): ("Температурный режим: {v}", "Температура: {v}"),
    ("temperature", "en", "list"): ("Temperature: {v}", "Temp. range: {v}"),
    ("temperature", "ru", "slang"): ("режим {v}", "t {v}"),
    ("securing", "ru", "prose"): ("Крепление: {v}.", "Закрепить груз {i}.",
                                  "Крепление груза {i}."),
    ("securing", "en", "prose"): ("Securing: {v}.", "Secure the cargo with {v}.",
                                  "Tie-down with {v} is required."),
    ("securing", "ru", "list"): ("Крепление: {v}", "Крепление груза: {v}"),
    ("securing", "en", "list"): ("Securing: {v}", "Tie-down: {v}"),
    ("securing", "ru", "slang"): ("крепёж: {v}", "крепить {i}"),
    ("packaging", "ru", "prose"): ("Упаковка: {v}.", "Груз {i}.", "Груз упакован {i}."),
    ("packaging", "en", "prose"): ("Packaging: {v}.", "The cargo is packed in {v}."),
    ("packaging", "ru", "list"): ("Упаковка: {v}",),
    ("packaging", "en", "list"): ("Packaging: {v}",),
    ("packaging", "ru", "slang"): ("упаковка: {v}",),
    ("oversize", "ru", "prose"): ("Негабарит: {v}.", "Габариты груза: {v}.",
                                  "Груз негабаритный: {v}."),
    ("oversize", "en", "prose"): ("Oversize cargo: {v}.", "Out-of-gauge dimensions: {v}.",
                                  "Cargo dimensions: {v}."),
    ("oversize", "ru", "list"): ("Габариты: {v}", "Негабарит: {v}"),
    ("oversize", "en", "list"): ("Dimensions: {v}", "Oversize: {v}"),
    ("oversize", "ru", "slang"): ("габариты {v}", "негабарит {v}"),
    ("sensors", "ru", "prose"): ("Нужен контроль {v}.", "Нужны датчики {v}.",
                                 "Обязателен мониторинг {v}."),
    ("sensors", "en", "prose"): ("Sensors required: {v}.", "Monitoring of {v} is required."),
    ("sensors", "ru", "list"): ("Датчики: контроль {v}", "Требуется контроль {v}"),
    ("sensors", "en", "list"): ("Sensors: {v}", "Monitoring: {v}"),
    ("sensors", "ru", "slang"): ("датчики {v}", "контроль {v}"),
}  # fmt: skip
# A temperature sensor alone may be written as a logger: (sentence, evidence).
_LOGGER: Final[dict[tuple[Language, ConditionStyle], tuple[tuple[str, str], ...]]] = {
    ("ru", "prose"): (("Нужен термописец.", "термописец"), ("С термописцем.", "термописцем")),
    ("en", "prose"): (("A temperature logger is required.", "temperature logger"),),
    ("ru", "list"): (("Термописец: нужен", "Термописец"),),
    ("en", "list"): (("Temperature logger: required", "Temperature logger"),),
    ("ru", "slang"): (("с термописцем", "термописцем"),),
}
# Named without values: (sentence, evidence); no word of a value list inside.
_WITHOUT_VALUES: Final[dict[tuple[str, Language, ConditionStyle], tuple[tuple[str, str], ...]]] = {
    ("temperature", "ru", "prose"): (("Нужен температурный режим.", "температурный режим"),
                                     ("Груз требует температурного режима.",
                                      "температурного режима")),
    ("temperature", "en", "prose"): (("A temperature-controlled trailer is needed.",
                                      "temperature-controlled"),),
    ("temperature", "ru", "list"): (("Температурный режим: уточняется", "Температурный режим"),),
    ("temperature", "en", "list"): (("Temperature: TBC", "Temperature"),),
    ("temperature", "ru", "slang"): (("нужен режим", "режим"),),
    ("securing", "ru", "prose"): (("Нужно крепление груза.", "крепление груза"),
                                  ("Груз необходимо закрепить.", "закрепить")),
    ("securing", "en", "prose"): (("The cargo must be secured.", "secured"),
                                  ("Securing of the cargo is required.", "Securing")),
    ("securing", "ru", "list"): (("Крепление: требуется", "Крепление"),),
    ("securing", "en", "list"): (("Securing: required", "Securing"),),
    ("securing", "ru", "slang"): (("нужен крепёж", "крепёж"),),
    ("packaging", "ru", "prose"): (("Нужна специальная упаковка.", "специальная упаковка"),
                                   ("Груз требует особой упаковки.", "особой упаковки")),
    ("packaging", "en", "prose"): (("Special packaging is required.", "Special packaging"),),
    ("packaging", "ru", "list"): (("Упаковка: особая", "Упаковка"),),
    ("packaging", "en", "list"): (("Packaging: special", "Packaging"),),
    ("packaging", "ru", "slang"): (("нужна упаковка", "упаковка"),),
    ("oversize", "ru", "prose"): (("Груз негабаритный.", "негабаритный"),
                                  ("Перевозка негабаритного груза.", "негабаритного")),
    ("oversize", "en", "prose"): (("The cargo is oversized.", "oversized"),
                                  ("This is an out-of-gauge load.", "out-of-gauge")),
    ("oversize", "ru", "list"): (("Негабарит: да", "Негабарит"),),
    ("oversize", "en", "list"): (("Oversize: yes", "Oversize"),),
    ("oversize", "ru", "slang"): (("негабарит", "негабарит"),),
    ("sensors", "ru", "prose"): (("Нужны датчики.", "датчики"),
                                 ("Нужен мониторинг груза датчиками.", "датчиками")),
    ("sensors", "en", "prose"): (("Sensor monitoring is required.", "Sensor monitoring"),),
    ("sensors", "ru", "list"): (("Датчики: нужны", "Датчики"),),
    ("sensors", "en", "list"): (("Sensors: required", "Sensors"),),
    ("sensors", "ru", "slang"): (("нужны датчики", "датчики"),),
}  # fmt: skip


def _style(lang: Language, style: ConditionStyle) -> ConditionStyle:
    return "prose" if style == "slang" and lang == "en" else style  # no English slang family


def _join(words: Sequence[str], lang: Language) -> str:
    if len(words) == 1:
        return words[0]
    last = " и " if lang == "ru" else " and "
    return ", ".join(words[:-1]) + last + words[-1]


def _signed(value: float, lang: Language, rng: Random, minus: str) -> str:
    number = normalize_number(value)
    digits = format_number(abs(number), lang, rng)
    if number > 0:
        return "+" + digits
    return minus + digits if number < 0 else digits


def temperature_text(condition: TemperatureCondition, lang: Language, rng: Random) -> str:
    """«+2…+6 °C», «от +2 до +6 °C», «не выше -18 °C», «between +2 and +6 °C»; a negative
    number takes a Unicode minus or a hyphen, the same one within a phrase."""
    low, high = condition.min_c, condition.max_c
    degree = rng.choice((" °C", "°C")) if lang == "ru" else " °C"
    minus = rng.choice(("\u2212", "-")) if lang == "ru" else "-"
    if low is not None and high is not None:
        lo, hi = _signed(low, lang, rng, minus), _signed(high, lang, rng, minus)
        if low == high:
            return f"{lo}{degree}"
        forms = ((f"{lo}…{hi}{degree}", f"от {lo} до {hi}{degree}", f"{lo}...{hi}{degree}")
                 if lang == "ru" else
                 (f"{lo} to {hi}{degree}", f"between {lo} and {hi}{degree}",
                  f"{lo}…{hi}{degree}"))  # fmt: skip
        return rng.choice(forms)
    if high is not None:
        upper: tuple[str, ...] = (("не выше", "не более") if lang == "ru" else
                                  ("not above", "no warmer than", "max"))  # fmt: skip
        return f"{rng.choice(upper)} {_signed(high, lang, rng, minus)}{degree}"
    assert low is not None
    lower: tuple[str, ...] = ("не ниже",) if lang == "ru" else ("not below", "min")
    return f"{rng.choice(lower)} {_signed(low, lang, rng, minus)}{degree}"


def rendered_length(metres: float, unit: Literal["m", "cm"]) -> Length:
    """The length as written: metres with one decimal («8,8 м», «6 м») or whole centimetres."""
    if unit == "cm":
        return Length(value=round(metres * 100), unit="cm")
    return Length(value=normalize_number(round(metres, 1)), unit="m")


def _length_number(value: float, lang: Language) -> str:
    """Lengths are never grouped by thousands: «1380 см», «8,8 м»."""
    text = str(value) if isinstance(value, int) else f"{value:.1f}"
    return text.replace(".", ",") if lang == "ru" else text


def _dimensions_text(dims: dict[Dimension, Length], lang: Language, rng: Random) -> str:
    unit = {"m": "м", "cm": "см"} if lang == "ru" else {"m": "m", "cm": "cm"}
    labels = ({"length": "длина", "width": "ширина", "height": "высота"} if lang == "ru" else
              {"length": "length", "width": "width", "height": "height"})  # fmt: skip
    numbers = {name: _length_number(d.value, lang) for name, d in dims.items()}
    one_unit = unit[next(iter(dims.values())).unit]
    if len(dims) == 3 and rng.random() < 0.4:
        a, b, c = (numbers[name] for name in _DIMENSIONS)
        if lang == "ru":
            return f"Д×Ш×В {a}×{b}×{c} {one_unit}"
        return f"L x W x H {a} x {b} x {c} {one_unit}"
    return ", ".join(f"{labels[name]} {numbers[name]} {one_unit}" for name in dims)


def _values_fragment(
    kind: str, text: str, lang: Language, style: ConditionStyle, rng: Random,
    gold: SpecialCondition, instrumental: str | None = None,
) -> Fragment:  # fmt: skip
    frame = rng.choice(_WITH_VALUES[(kind, lang, style)])
    sentence = frame.format(v=text, i=instrumental if instrumental is not None else text)
    evidence = instrumental if "{i}" in frame and instrumental is not None else text
    return Fragment(sentence, (RenderedField(text=evidence, field="special_conditions",
                                             gold_value=gold),),
                    frozenset({f"condition:{kind}"}))  # fmt: skip


def _empty_fragment(kind: str, lang: Language, style: ConditionStyle, rng: Random,
                    gold: SpecialCondition) -> Fragment:  # fmt: skip
    sentence, evidence = rng.choice(_WITHOUT_VALUES[(kind, lang, style)])
    return Fragment(sentence, (RenderedField(text=evidence, field="special_conditions",
                                             gold_value=gold),),
                    frozenset({f"condition:{kind}"}))  # fmt: skip


def _temperature(c: TemperatureCondition, draft: RequestDraft, lang: Language,
                 style: ConditionStyle, rng: Random, empty: bool) -> Fragment:  # fmt: skip
    if empty:
        gold = TemperatureCondition(kind="temperature", min_c=None, max_c=None)
        return _empty_fragment("temperature", lang, style, rng, gold)
    return _values_fragment("temperature", temperature_text(c, lang, rng), lang, style, rng, c)


def _securing(c: SecuringCondition, draft: RequestDraft, lang: Language, style: ConditionStyle,
              rng: Random, empty: bool) -> Fragment:  # fmt: skip
    if empty:
        return _empty_fragment("securing", lang, style, rng,
                               SecuringCondition(kind="securing", methods=[]))  # fmt: skip
    if lang == "ru":
        nominative = _join([RU_SECURING[m][0] for m in c.methods], lang)
        instrumental = _join([RU_SECURING[m][1] for m in c.methods], lang)
        return _values_fragment("securing", nominative, lang, style, rng, c, instrumental)
    return _values_fragment("securing", _join([EN_SECURING[m] for m in c.methods], lang), lang,
                            style, rng, c)  # fmt: skip


def _packaging(c: PackagingCondition, draft: RequestDraft, lang: Language,
               style: ConditionStyle, rng: Random, empty: bool) -> Fragment:  # fmt: skip
    if empty:
        return _empty_fragment("packaging", lang, style, rng,
                               PackagingCondition(kind="packaging", types=[]))  # fmt: skip
    if lang == "ru":
        nominative = _join([RU_PACKAGING[t][0] for t in c.types], lang)
        prepositional = _join([RU_PACKAGING[t][1] for t in c.types], lang)
        return _values_fragment("packaging", nominative, lang, style, rng, c, prepositional)
    return _values_fragment("packaging", _join([EN_PACKAGING[t] for t in c.types], lang), lang,
                            style, rng, c)  # fmt: skip


def _oversize(c: OversizeCondition, draft: RequestDraft, lang: Language, style: ConditionStyle,
              rng: Random, empty: bool) -> Fragment:  # fmt: skip
    if empty:
        gold = OversizeCondition(kind="oversize", length=None, width=None, height=None)
        return _empty_fragment("oversize", lang, style, rng, gold)
    kept = set(draft.oversize_dims) if draft.oversize_dims is not None else set(_DIMENSIONS)
    dims: dict[Dimension, Length] = {}
    for name in _DIMENSIONS:
        value = getattr(c, name)
        if value is not None and name in kept:
            dims[name] = rendered_length(float(value.value), draft.length_unit)
    gold = OversizeCondition(kind="oversize", length=dims.get("length"), width=dims.get("width"),
                             height=dims.get("height"))  # fmt: skip
    return _values_fragment("oversize", _dimensions_text(dims, lang, rng), lang, style, rng, gold)


def _sensors(c: SensorsCondition, draft: RequestDraft, lang: Language, style: ConditionStyle,
             rng: Random, empty: bool) -> Fragment:  # fmt: skip
    if empty:
        return _empty_fragment("sensors", lang, style, rng,
                               SensorsCondition(kind="sensors", parameters=[]))  # fmt: skip
    if c.parameters == ["temperature"] and rng.random() < 0.4:
        sentence, evidence = rng.choice(_LOGGER[(lang, style)])
        return Fragment(sentence, (RenderedField(text=evidence, field="special_conditions",
                                                 gold_value=c),),
                        frozenset({"condition:sensors"}))  # fmt: skip
    words = RU_SENSORS if lang == "ru" else EN_SENSORS
    return _values_fragment("sensors", _join([words[p] for p in c.parameters], lang), lang,
                            style, rng, c)  # fmt: skip


_RENDERERS: Final[dict[str, Callable[..., Fragment]]] = {
    "temperature": _temperature, "securing": _securing, "packaging": _packaging,
    "oversize": _oversize, "sensors": _sensors,
}  # fmt: skip


def condition_fragments(
    facts: LoadFactsV2, draft: RequestDraft, rng: Random, style: ConditionStyle
) -> dict[str, Fragment]:
    """`cond_<kind>` -> fragment for every condition the draft puts into the text."""
    lang: Language = draft.language
    chosen = _style(lang, style)
    fragments: dict[str, Fragment] = {}
    for condition in facts.special_conditions:
        if condition.kind in draft.conditions_dropped:
            continue
        empty = condition.kind in draft.conditions_without_values
        fragments[CONDITION_PREFIX + condition.kind] = _RENDERERS[condition.kind](
            condition, draft, lang, chosen, rng, empty)  # fmt: skip
    return fragments
