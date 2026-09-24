"""Every word and phrase of the request generator, in one module (step 6 refactoring rule).

Russian nouns come as (one, few, many) forms: 1 паллета, 2 паллеты, 5 паллет. The card_v2
words (sub-step V3) sit next to the card_v1 ones, which do not change: `VOCABULARIES` picks the
set of the schema being rendered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qf.contracts import (
    CargoCategory,
    CargoCategoryV2,
    EquipmentType,
    EquipmentTypeV2,
    Language,
    PackagingType,
    SecuringMethod,
    SensorParameter,
)

__all__ = [
    "CATEGORY_WORDS",
    "CATEGORY_WORDS_V2",
    "EN_MACHINERY_PIECE_NOUNS",
    "EN_PACKAGING",
    "EN_SECURING",
    "EN_SENSORS",
    "EQUIPMENT_WORDS_SLANG_V2",
    "EQUIPMENT_WORDS_V2",
    "RU_MACHINERY_PIECE_NOUNS",
    "RU_MACHINERY_PIECE_SLANG",
    "RU_PACKAGING",
    "RU_SECURING",
    "RU_SENSORS",
    "VOCABULARIES",
    "CardVocabulary",
    "EN_MONTHS",
    "EN_MONTHS_SHORT",
    "EN_PIECE_NOUNS",
    "EN_TONNE_WORDS",
    "EQUIPMENT_WORDS",
    "EQUIPMENT_WORDS_SLANG",
    "RELATIVE_DAYS",
    "RU_MONTHS_GENITIVE",
    "RU_PIECE_NOUNS",
    "RU_PIECE_SLANG",
    "RU_TONNE",
    "RuForms",
    "ru_plural",
]

RuForms = tuple[str, str, str]

RU_MONTHS_GENITIVE: Final = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)  # fmt: skip
EN_MONTHS: Final = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
EN_MONTHS_SHORT: Final = tuple(month[:3] for month in EN_MONTHS)

RELATIVE_DAYS: Final[dict[Language, tuple[str, str, str]]] = {
    "ru": ("сегодня", "завтра", "послезавтра"),
    "en": ("today", "tomorrow", "the day after tomorrow"),
}

RU_PIECE_NOUNS: Final[tuple[RuForms, ...]] = (
    ("место", "места", "мест"),
    ("паллета", "паллеты", "паллет"),
    ("грузоместо", "грузоместа", "грузомест"),
    ("коробка", "коробки", "коробок"),
    ("поддон", "поддона", "поддонов"),
)
RU_PIECE_SLANG: Final = ("палл.", "пал.", "шт.")  # abbreviations: the same form for any number
EN_PIECE_NOUNS: Final = (("pallet", "pallets"), ("piece", "pieces"), ("package", "packages"))
RU_TONNE: Final[RuForms] = ("тонна", "тонны", "тонн")
EN_TONNE_WORDS: Final = ("t", "tonnes", "metric tons")

EQUIPMENT_WORDS: Final[dict[Language, dict[EquipmentType, tuple[str, ...]]]] = {
    "ru": {
        "dry_van": ("тент", "тентованный полуприцеп", "фургон", "сухой фургон"),
        "reefer": ("рефрижератор", "реф", "рефрижераторный полуприцеп"),
    },
    "en": {
        "dry_van": ("dry van", "box trailer", "dry van trailer"),
        "reefer": ("reefer", "refrigerated trailer", "reefer trailer"),
    },
}
EQUIPMENT_WORDS_SLANG: Final[dict[EquipmentType, tuple[str, ...]]] = {
    "dry_van": ("тент", "тентовоз"),
    "reefer": ("реф", "рефрижератор"),
}

CATEGORY_WORDS: Final[dict[Language, dict[CargoCategory, tuple[str, ...]]]] = {
    "ru": {
        "general": ("генеральный груз",),
        "retail": ("товары для розничных сетей", "товары для ритейла"),
        "consumer_goods": ("товары народного потребления", "ТНП"),
        "food_beverage": ("продукты питания", "продукты и напитки"),
        "automotive": ("автозапчасти", "автокомпоненты"),
        "electronics": ("электроника", "бытовая электроника"),
    },
    "en": {
        "general": ("general cargo", "general freight"),
        "retail": ("retail goods", "retail merchandise"),
        "consumer_goods": ("consumer goods",),
        "food_beverage": ("food and beverages", "food products"),
        "automotive": ("auto parts", "automotive parts"),
        "electronics": ("electronics", "electronic equipment"),
    },
}


# --- card_v2 (sub-step V3) ------------------------------------------------------------------

EQUIPMENT_WORDS_V2: Final[dict[Language, dict[EquipmentTypeV2, tuple[str, ...]]]] = {
    "ru": {
        "tent": ("тент", "тентованный полуприцеп", "фура", "шторный полуприцеп"),
        "van": ("цельнометаллический фургон", "фургон", "цельномет"),
        "reefer": ("рефрижератор", "реф", "рефрижераторный полуприцеп"),
        "isotherm": ("изотерм", "изотермический фургон"),
        "container": ("контейнеровоз", "машина под контейнер"),
        "lowbed": ("трал", "низкорамный трал", "низкорамный полуприцеп"),
        "mega": ("мега", "мега-тент", "тент мега"),
        "flatbed": ("бортовая платформа", "бортовой полуприцеп", "площадка"),
    },
    "en": {
        "tent": ("curtainsider", "tautliner", "tilt trailer"),
        "van": ("box van", "box trailer", "dry van"),
        "reefer": ("reefer", "refrigerated trailer", "reefer trailer"),
        "isotherm": ("insulated trailer", "insulated van"),
        "container": ("container truck", "container chassis"),
        "lowbed": ("lowboy", "low-bed trailer", "lowbed"),
        "mega": ("mega trailer", "mega curtainsider"),
        "flatbed": ("flatbed", "flatbed trailer"),
    },
}
EQUIPMENT_WORDS_SLANG_V2: Final[dict[EquipmentTypeV2, tuple[str, ...]]] = {
    "tent": ("тент", "фура"),
    "van": ("цельномет", "фургон"),
    "reefer": ("реф", "рефрижератор"),
    "isotherm": ("изотерм",),
    "container": ("контейнеровоз",),
    "lowbed": ("трал",),
    "mega": ("мега",),
    "flatbed": ("площадка", "борт"),
}
CATEGORY_WORDS_V2: Final[dict[Language, dict[CargoCategoryV2, tuple[str, ...]]]] = {
    "ru": {
        "general": CATEGORY_WORDS["ru"]["general"],
        "retail": CATEGORY_WORDS["ru"]["retail"],
        "consumer_goods": CATEGORY_WORDS["ru"]["consumer_goods"],
        "food_beverage": CATEGORY_WORDS["ru"]["food_beverage"],
        "automotive": CATEGORY_WORDS["ru"]["automotive"],
        "electronics": CATEGORY_WORDS["ru"]["electronics"],
        "machinery": ("спецтехника", "спецтехника и оборудование", "строительная техника"),
    },
    "en": {
        "general": CATEGORY_WORDS["en"]["general"],
        "retail": CATEGORY_WORDS["en"]["retail"],
        "consumer_goods": CATEGORY_WORDS["en"]["consumer_goods"],
        "food_beverage": CATEGORY_WORDS["en"]["food_beverage"],
        "automotive": CATEGORY_WORDS["en"]["automotive"],
        "electronics": CATEGORY_WORDS["en"]["electronics"],
        "machinery": ("heavy machinery", "construction equipment"),
    },
}
RU_MACHINERY_PIECE_NOUNS: Final[tuple[RuForms, ...]] = (
    ("единица техники", "единицы техники", "единиц техники"),
    ("единица", "единицы", "единиц"),  # not «машина»: in a request that reads as a truck
)
RU_MACHINERY_PIECE_SLANG: Final = ("ед.", "шт.")
EN_MACHINERY_PIECE_NOUNS: Final = (("unit", "units"), ("machine", "machines"))

# Conditions: (nominative, instrumental) for securing; (after «Упаковка:», «в …») for packaging;
# genitive («контроль …») for sensors. No phrase of a value list appears in the phrases that
# name a condition without values (`conditions.py`).
RU_SECURING: Final[dict[SecuringMethod, tuple[str, str]]] = {
    "straps": ("ремни", "ремнями"),
    "chains": ("цепи", "цепями"),
    "wheel_chocks": ("противооткатные упоры", "противооткатными упорами"),
    "anti_slip_mats": ("антискользящие коврики", "антискользящими ковриками"),
    "load_bars": ("распорные штанги", "распорными штангами"),
}
EN_SECURING: Final[dict[SecuringMethod, str]] = {
    "straps": "straps", "chains": "chains", "wheel_chocks": "wheel chocks",
    "anti_slip_mats": "anti-slip mats", "load_bars": "load bars",
}  # fmt: skip
RU_PACKAGING: Final[dict[PackagingType, tuple[str, str]]] = {
    "crate": ("деревянная обрешётка", "в деревянной обрешётке"),
    "stretch_film": ("стрейч-плёнка", "в стрейч-плёнке"),
    "moisture_protection": ("влагозащитная", "во влагозащитной упаковке"),
    "shock_protection": ("амортизирующая", "в амортизирующей упаковке"),
}
EN_PACKAGING: Final[dict[PackagingType, str]] = {
    "crate": "wooden crates", "stretch_film": "stretch film",
    "moisture_protection": "moisture-proof packaging",
    "shock_protection": "shock-absorbing packaging",
}  # fmt: skip
RU_SENSORS: Final[dict[SensorParameter, str]] = {
    "temperature": "температуры", "humidity": "влажности", "pressure": "давления",
    "tilt": "наклона", "shock": "ударов", "door_opening": "открытия дверей",
}  # fmt: skip
EN_SENSORS: Final[dict[SensorParameter, str]] = {
    "temperature": "temperature", "humidity": "humidity", "pressure": "pressure",
    "tilt": "tilt", "shock": "shock", "door_opening": "door opening",
}  # fmt: skip


@dataclass(frozen=True)
class CardVocabulary:
    """The words of one answer schema that depend on its value lists."""

    # keys are the Literal values of the schema's equipment types and cargo categories
    equipment: Mapping[Language, Mapping[Any, tuple[str, ...]]]
    equipment_slang: Mapping[Any, tuple[str, ...]]
    category: Mapping[Language, Mapping[Any, tuple[str, ...]]]


VOCABULARIES: Final[dict[str, CardVocabulary]] = {
    "card_v1": CardVocabulary(EQUIPMENT_WORDS, EQUIPMENT_WORDS_SLANG, CATEGORY_WORDS),
    "card_v2": CardVocabulary(EQUIPMENT_WORDS_V2, EQUIPMENT_WORDS_SLANG_V2, CATEGORY_WORDS_V2),
}


def ru_plural(n: int, forms: RuForms) -> str:
    """Russian noun form for an integer count: 1 паллета, 2 паллеты, 5 паллет, 11 паллет."""
    if n % 100 in (11, 12, 13, 14):
        return forms[2]
    if n % 10 == 1:
        return forms[0]
    if n % 10 in (2, 3, 4):
        return forms[1]
    return forms[2]
