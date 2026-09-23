"""Every word and phrase of the request generator, in one module (step 6 refactoring rule).

Russian nouns come as (one, few, many) forms: 1 паллета, 2 паллеты, 5 паллет.
"""

from __future__ import annotations

from typing import Final

from qf.contracts import CargoCategory, EquipmentType, Language

__all__ = [
    "CATEGORY_WORDS",
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


def ru_plural(n: int, forms: RuForms) -> str:
    """Russian noun form for an integer count: 1 паллета, 2 паллеты, 5 паллет, 11 паллет."""
    if n % 100 in (11, 12, 13, 14):
        return forms[2]
    if n % 10 == 1:
        return forms[0]
    if n % 10 in (2, 3, 4):
        return forms[1]
    return forms[2]
