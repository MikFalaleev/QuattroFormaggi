"""A simple rule-based extractor (D-109): regular expressions and keyword lists, no model.

It is the lower bound an LLM has to beat, written the way a developer would write a first
extractor: its own word lists (not the generator's), the city catalog as a gazetteer, the
nearest cue word to tell origin from destination and pickup from delivery. The rules were
written looking at `generated_v2/val` only, never at a benchmark. It never reports conflicts
and never guesses what the text does not say.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Final

from qf.domain import CITIES

__all__ = ["RULES_VERSION", "extract_card"]

RULES_VERSION: Final = "rules_v1"

_FLAGS: Final = re.IGNORECASE
_NUM: Final = r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?"
_SIGNED: Final = r"[+\-−]?\s?\d+(?:[.,]\d+)?"

# --- places ---------------------------------------------------------------------------------

_CITY_FORMS: Final = sorted(
    ((form, name) for name, info in CITIES.items()
     for form in {name, info.name_en, info.genitive, info.accusative}),
    key=lambda pair: -len(pair[0]),
)  # fmt: skip
_CITY_RE: Final = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(form) for form, _ in _CITY_FORMS) + r")(?![\w-])", _FLAGS
)
_CITY_BY_FORM: Final = {form.casefold(): name for form, name in _CITY_FORMS}
_ORIGIN_CUES: Final = re.compile(
    r"(?:\bиз\b|\bот\b|откуда|пункт\s+загрузки|место\s+загрузки|загрузк\w*|погрузк\w*|"
    r"\bfrom\b|origin|pick-?\s?up|loading)[^\n]{0,12}$", _FLAGS)  # fmt: skip
_DEST_CUES: Final = re.compile(
    r"(?:\bв\b|\bво\b|\bдо\b|куда|пункт\s+(?:выгрузки|разгрузки|доставки)|место\s+(?:выгрузки|"
    r"доставки)|выгрузк\w*|доставк\w*|\bto\b|destination|delivery|drop|unloading)[^\n]{0,12}$",
    _FLAGS)  # fmt: skip


def _places(text: str) -> tuple[str | None, str | None]:
    mentions = []
    for match in _CITY_RE.finditer(text):
        before = text[max(0, match.start() - 30) : match.start()]
        role = ("dest" if _DEST_CUES.search(before) else
                "origin" if _ORIGIN_CUES.search(before) else None)  # fmt: skip
        mentions.append((_CITY_BY_FORM[match.group(1).casefold()], role))
    origin = next((c for c, role in mentions if role == "origin"), None)
    destination = next((c for c, role in mentions if role == "dest" and c != origin), None)
    unlabeled = [c for c, role in mentions if role is None]
    if origin is None:
        origin = next((c for c in unlabeled + [c for c, _ in mentions] if c != destination), None)
    if destination is None:
        destination = next((c for c in unlabeled + [c for c, _ in mentions] if c != origin), None)
    return origin, destination


# --- dates ----------------------------------------------------------------------------------

_MONTHS_RU: Final = ("январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр",
                     "октябр", "ноябр", "декабр")  # fmt: skip
_MONTHS_EN: Final = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                     "dec")  # fmt: skip
_DATE_RES: Final = (
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("dmy", re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")),
    ("ru", re.compile(r"\b(\d{1,2})\s+(январ[яь]|феврал[яь]|марта?|апрел[яь]|ма[яй]|июн[яь]|"
                      r"июл[яь]|августа?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])"
                      r"(?:\s+(\d{4}))?", _FLAGS)),
    ("en_mdy", re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
                          r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", _FLAGS)),
    ("en_dmy", re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                          r"[a-z]*\.?,?\s+(\d{4})", _FLAGS)),
    ("us", re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")),
    ("rel", re.compile(r"\b(послезавтра|завтра|сегодня|the day after tomorrow|tomorrow|today)\b",
                       _FLAGS)),
)  # fmt: skip
_RELATIVE: Final = {"сегодня": 0, "today": 0, "завтра": 1, "tomorrow": 1, "послезавтра": 2,
                    "the day after tomorrow": 2}  # fmt: skip
_DELIVERY_CUES: Final = re.compile(
    r"(?:выгруз|разгруз|доставк|доставить|сдать|прибыт|delivery|deliver|unload|drop|arriv)"
    r"[^\n]{0,20}$", _FLAGS)  # fmt: skip


def _month(word: str, table: tuple[str, ...]) -> int:
    word = word.casefold()
    for number, stem in enumerate(table, start=1):
        if word.startswith(stem) and (stem != "ма" or word in ("мая", "май")):
            return number
    raise ValueError(word)


def _date_of(kind: str, match: re.Match[str], today: date | None) -> date | None:
    g = match.groups()
    try:
        if kind == "iso":
            return date(int(g[0]), int(g[1]), int(g[2]))
        if kind == "dmy":
            return date(int(g[2]), int(g[1]), int(g[0]))
        if kind == "ru":
            year = int(g[2]) if g[2] else (today.year if today else None)
            return date(year, _month(g[1], _MONTHS_RU), int(g[0])) if year else None
        if kind == "en_mdy":
            return date(int(g[2]), _month(g[0][:3], _MONTHS_EN), int(g[1]))
        if kind == "en_dmy":
            return date(int(g[2]), _month(g[1][:3], _MONTHS_EN), int(g[0]))
        if kind == "us":
            return date(int(g[2]), int(g[0]), int(g[1]))
        return today + timedelta(days=_RELATIVE[g[0].casefold()]) if today else None
    except ValueError:
        return None


def _dates(text: str, today: date | None) -> tuple[date | None, date | None]:
    found: list[tuple[int, int, date, bool]] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _DATE_RES:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            value = _date_of(kind, match, today)
            if value is None:
                continue
            taken.append(span)
            cue = bool(_DELIVERY_CUES.search(text[max(0, span[0] - 30) : span[0]]))
            found.append((span[0], span[1], value, cue))
    found.sort()
    pickup = next((d for _, _, d, is_delivery in found if not is_delivery), None)
    delivery = next((d for _, _, d, is_delivery in found if is_delivery), None)
    if delivery is None:
        rest = [d for _, _, d, _ in found]
        if pickup in rest:
            rest.remove(pickup)
        delivery = rest[0] if rest else None
    return pickup, delivery


# --- weight and pieces ----------------------------------------------------------------------

_WEIGHT_RE: Final = re.compile(
    rf"(?<![\d.,])({_NUM})\s*(metric\s+tons?|tonnes?|tons?|тонн[аы]?|тн|т|t|кг|kg)(?![\w])",
    _FLAGS)  # fmt: skip
_PER_PIECE_AFTER: Final = re.compile(
    r"^\s*(?:each|per\s+(?:pallet|piece|unit|box)|/\s*(?:место|паллет|шт)|на\s+(?:место|паллету|"
    r"поддон|единицу)|за\s+место|каждое|каждая|каждый)", _FLAGS)  # fmt: skip
_PIECES_RE: Final = re.compile(
    r"(?<![\d.,])(\d{1,4})\s*(паллет\w*|палл?\.?|пал\b|поддон\w*|мест[оа]?\b|коробо?к\w*|ящик\w*|"
    r"грузомест\w*|единиц\w*|ед\.|шт\.?|упаков\w*|пач\w*|рул\w*|pallets?|pieces?|pcs|boxes|box|"
    r"cartons?|units?|bundles?|crates?|packages?)", _FLAGS)  # fmt: skip


def _number(text: str, unit_is_kg: bool) -> float:
    raw = text.replace(" ", "").replace(" ", "")
    if "," in raw and "." not in raw:
        head, _, tail = raw.rpartition(",")
        raw = head + tail if unit_is_kg and len(tail) == 3 else f"{head}.{tail}"
    return float(raw.replace(",", ""))


def _weights(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    total = per_piece = None
    for match in _WEIGHT_RE.finditer(text):
        unit = "kg" if match.group(2).casefold() in ("кг", "kg") else "t"
        value = _number(match.group(1), unit == "kg")
        if value <= 0:
            continue
        quantity = {"value": int(value) if value.is_integer() else value, "unit": unit}
        before = text[max(0, match.start() - 4) : match.start()].casefold()
        if before.endswith("по ") or _PER_PIECE_AFTER.search(text[match.end() : match.end() + 20]):
            per_piece = per_piece or quantity
        elif total is None:
            total = quantity
    return total, per_piece


def _pieces(text: str) -> int | None:
    for match in _PIECES_RE.finditer(text):
        if text[max(0, match.start() - 3) : match.start()].casefold().endswith("по "):
            continue
        count = int(match.group(1))
        if count > 0:
            return count
    if re.search(r"\b(?:one|одна|один)\s+(?:unit|единица|машина|экскаватор|wheel loader|piece)",
                 text, _FLAGS):  # fmt: skip
        return 1
    return None


# --- what, who, which truck ------------------------------------------------------------------

_EQUIPMENT: Final = (
    ("mega", r"мега|mega"), ("isotherm", r"изотерм|insulated"),
    ("reefer", r"рефриж|\bреф\b|reefer|refrigerat"), ("lowbed", r"трал|низкорам|lowboy|low-?bed"),
    ("container", r"контейнер|container"), ("flatbed", r"платформ|площадк|бортов|\bборт\b|flatbed"),
    ("van", r"цельномет|фургон|box van|box trailer|dry van"),
    ("tent", r"тент|фур[аыу]\b|штор|curtain|tautliner|tilt"),
)  # fmt: skip
_CATEGORY: Final = (
    ("machinery", r"спецтехник|техник[аи]\b|экскаватор|бульдозер|автокран|каток|погрузчик|"
                  r"machinery|construction equipment|loader|excavator"),
    ("food_beverage", r"продукт|напит|молоч|кондитер|шоколад|овощ|фрукт|консерв|пряник|"
                      r"food|beverage|dairy"),
    ("electronics", r"электрон|electronic"),
    ("automotive", r"автозапчаст|автокомпонент|auto parts|automotive"),
    ("consumer_goods", r"народного потребления|\bтнп\b|consumer goods"),
    ("retail", r"ритейл|розничн|retail"),
    ("general", r"генеральн|general cargo|general freight"),
)  # fmt: skip
_SHIPPER_RE: Final = re.compile(
    r"(?i:грузоотправитель|отправитель|shipper|клиент|client|customer|компани[яи]|company|"
    r"on behalf of|\bот\b|\bfor\b|\bдля\b)"
    r"[ \t]*[:—–-]?[ \t]*(ООО[ \t]*«[^»\n]+»|АО[ \t]*«[^»\n]+»|"
    r"[A-Z][\w&.'-]*(?:[ \t]+[A-Z][\w&.'-]*){0,3})"
)
_CITY_NAMES_EN: Final = {info.name_en.casefold() for info in CITIES.values()}


def _first(table: tuple[tuple[str, str], ...], text: str) -> str | None:
    return next((name for name, pattern in table if re.search(pattern, text, _FLAGS)), None)


def _shipper(text: str) -> str | None:
    for match in _SHIPPER_RE.finditer(text):
        name = match.group(1).strip(" .,;:")
        words = name.split()
        if name.casefold() in _CITY_NAMES_EN or words[0].casefold() in _CITY_NAMES_EN:
            continue
        if len(words) == 1 and not name.startswith(("ООО", "АО")):
            continue  # a single capitalized word is too often not a company
        return name
    return None


# --- special conditions ---------------------------------------------------------------------

_SENTENCES: Final = re.compile(r"(?<=[.!?;])\s+|\n")
_TEMP_RANGE: Final = re.compile(
    rf"({_SIGNED})\s*(?:°\s*[CС])?\s*(?:…|\.{{2,3}}|–|—|-|\bдо\b|\bto\b|\band\b)\s*({_SIGNED})"
    r"\s*°?\s*[CС]?", _FLAGS)  # fmt: skip
_TEMP_MAX: Final = re.compile(rf"(?:не выше|не более|\bдо\b|not above|no higher than|"
                              rf"max(?:imum)?|up to)\s*({_SIGNED})\s*°?\s*[CС]?",
                              _FLAGS)  # fmt: skip
_TEMP_MIN: Final = re.compile(rf"(?:не ниже|не менее|\bот\b|not below|no lower than|"
                              rf"min(?:imum)?|at least)\s*({_SIGNED})\s*°?\s*[CС]?",
                              _FLAGS)  # fmt: skip
_TEMP_ONE: Final = re.compile(rf"({_SIGNED})\s*°\s*[CС]", _FLAGS)
_TEMP_CUE: Final = re.compile(r"°|температур|temperature|\bрежим|\btemp\b|термо", _FLAGS)
_SENSOR_CUE: Final = re.compile(r"датчик|контрол|мониторинг|термописц|термописец|logger|sensor|"
                                r"monitor|tracking|record", _FLAGS)  # fmt: skip
_SENSORS: Final = (("temperature", r"температур|temperature|термописц|термописец|logger"),
                   ("humidity", r"влажн|humidity"), ("pressure", r"давлен|pressure"),
                   ("tilt", r"наклон|горизонт|tilt"), ("shock", r"удар|shock"),
                   ("door_opening", r"двер|door"))  # fmt: skip
_SECURING: Final = (("straps", r"ремн|ремен|стяжк|strap"), ("chains", r"цеп[иья]|chain"),
                    ("wheel_chocks", r"упор|chock"), ("anti_slip_mats", r"коврик|anti-?slip|"
                     r"антискольз"), ("load_bars", r"штанг|load bar"))  # fmt: skip
_SECURING_CUE: Final = re.compile(r"крепл|крепит|закреп|securing|secure|lashing", _FLAGS)
_PACKAGING: Final = (("crate", r"обрешет|обрешёт|crate"), ("stretch_film", r"стрейч|stretch"),
                     ("moisture_protection", r"влагозащит|moisture|waterproof"),
                     ("shock_protection", r"амортиз|shock-?absorb|cushion"))  # fmt: skip
_PACKAGING_CUE: Final = re.compile(r"упаковк|packaging", _FLAGS)
_OVERSIZE_CUE: Final = re.compile(r"негабарит|oversize|out-of-gauge|габарит|dimension", _FLAGS)
_DIM_TRIPLE: Final = re.compile(rf"({_NUM})\s*[×xх*]\s*({_NUM})\s*[×xх*]\s*({_NUM})\s*(см|cm|м|m)"
                                r"\b", _FLAGS)  # fmt: skip
_DIM: Final = {
    "length": re.compile(rf"(?:длин\w*|length|\bL\b|\bД\b)\s*[:—-]?\s*({_NUM})\s*(см|cm|м|m)\b",
                         _FLAGS),
    "width": re.compile(rf"(?:ширин\w*|width|\bW\b|\bШ\b)\s*[:—-]?\s*({_NUM})\s*(см|cm|м|m)\b",
                        _FLAGS),
    "height": re.compile(rf"(?:высот\w*|height|\bH\b|\bВ\b)\s*[:—-]?\s*({_NUM})\s*(см|cm|м|m)\b",
                         _FLAGS),
}  # fmt: skip


def _signed(text: str) -> float:
    value = float(text.replace("−", "-").replace(" ", "").replace(",", "."))
    return int(value) if value.is_integer() else value


def _length(value: str, unit: str) -> dict[str, Any]:
    number = _number(value, unit_is_kg=False)
    return {"value": int(number) if number.is_integer() else number,
            "unit": "cm" if unit.casefold() in ("см", "cm") else "m"}  # fmt: skip


def _temperature(sentences: list[str]) -> dict[str, Any] | None:
    for sentence in sentences:
        if not _TEMP_CUE.search(sentence) or _SENSOR_CUE.search(sentence) and not re.search(
                r"°|режим|держать|keep|range", sentence, _FLAGS):  # fmt: skip
            continue
        if (match := _TEMP_RANGE.search(sentence)) and "°" in sentence:
            low, high = sorted((_signed(match.group(1)), _signed(match.group(2))))
            return {"kind": "temperature", "min_c": low, "max_c": high}
        if match := _TEMP_MAX.search(sentence):
            return {"kind": "temperature", "min_c": None, "max_c": _signed(match.group(1))}
        if match := _TEMP_MIN.search(sentence):
            return {"kind": "temperature", "min_c": _signed(match.group(1)), "max_c": None}
        if match := _TEMP_ONE.search(sentence):
            value = _signed(match.group(1))
            return {"kind": "temperature", "min_c": value, "max_c": value}
        if re.search(r"температурн\w* режим|temperature[- ]control|temperature regime", sentence,
                     _FLAGS):  # fmt: skip
            return {"kind": "temperature", "min_c": None, "max_c": None}
    return None


def _listed(table: tuple[tuple[str, str], ...], text: str) -> list[str]:
    return [name for name, pattern in table if re.search(pattern, text, _FLAGS)]


def _conditions(text: str) -> list[dict[str, Any]]:
    sentences = [s for s in _SENTENCES.split(text) if s.strip()]
    conditions = []
    if temperature := _temperature(sentences):
        conditions.append(temperature)
    securing = _listed(_SECURING, text)
    if securing or _SECURING_CUE.search(text):
        conditions.append({"kind": "securing", "methods": securing})
    packaging = _listed(_PACKAGING, text)
    if packaging or _PACKAGING_CUE.search(text):
        conditions.append({"kind": "packaging", "types": packaging})
    dims: dict[str, Any] = {"length": None, "width": None, "height": None}
    if match := _DIM_TRIPLE.search(text):
        for name, value in zip(("length", "width", "height"), match.groups()[:3], strict=True):
            dims[name] = _length(value, match.group(4))
    for name, pattern in _DIM.items():
        if dims[name] is None and (match := pattern.search(text)):
            dims[name] = _length(match.group(1), match.group(2))
    if any(dims.values()) or _OVERSIZE_CUE.search(text):
        conditions.append({"kind": "oversize", **dims})
    sensor_sentences = " ".join(s for s in sentences if _SENSOR_CUE.search(s))
    if sensor_sentences:
        conditions.append({"kind": "sensors", "parameters": _listed(_SENSORS, sensor_sentences)})
    return conditions


# --- the card -------------------------------------------------------------------------------


def extract_card(user_message: str, today: date | None) -> dict[str, Any]:
    """A card_v2 dict from the user message (the request date line included)."""
    text = user_message.split("\n\n", 1)[-1] if today is not None else user_message
    shipper = _shipper(text)
    without_shipper = text.replace(shipper, " ") if shipper else text
    origin, destination = _places(text)
    pickup, delivery = _dates(text, today)
    total, per_piece = _weights(text)
    return {
        "shipper_name": shipper,
        "cargo_category": _first(_CATEGORY, without_shipper),
        "equipment_type": _first(_EQUIPMENT, text),
        "pieces": _pieces(text),
        "weight_total": total,
        "weight_per_piece": per_piece if total is None else None,
        "origin": {"city": origin, "region": CITIES[origin].region} if origin else None,
        "destination": (
            {"city": destination, "region": CITIES[destination].region} if destination else None
        ),  # fmt: skip
        "pickup_date": pickup.isoformat() if pickup else None,
        "delivery_date": delivery.isoformat() if delivery else None,
        "special_conditions": _conditions(without_shipper),  # «Supply Chain» is no chain
    }
