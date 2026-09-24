"""Russian template families: T1 business letter, T2 messenger, T3 list, T7 slang (OOD only).

Families only compose fragments from `fields.py`; every slot has sentences for each subset of
its fields, so a dropped field removes its words instead of leaving a gap.
"""

from __future__ import annotations

from qf.data.registries import TEMPLATE_FAMILIES
from qf.data.render.base import LayoutFamily

__all__ = ["BusinessLetterRu", "ListRu", "MessengerRu", "SlangRu"]

_NOTES = {
    "weight_conflict": ("{weight_conflict}",),
    "pieces_conflict": ("{pieces_conflict}",),
    "distractors": ("{distractors}",),
}


@TEMPLATE_FAMILIES.register("T1")
class BusinessLetterRu(LayoutFamily):
    name = "T1"
    language = "ru"
    slots = {
        "hello": ("Добрый день!", "Здравствуйте!", "Добрый день, коллеги!",
                  "Здравствуйте, коллеги!"),
        "shipper": ("Пишем от компании {shipper}.", "Вас беспокоит компания {shipper}.",
                    "Компания {shipper} просит организовать перевозку."),
        "route": ("Просим рассчитать стоимость перевозки {origin_from} {destination_to}.",
                  "Необходимо перевезти груз {origin_from} {destination_to}.",
                  "Просим рассчитать стоимость перевозки {origin_from}.",
                  "Необходимо доставить груз {destination_to}."),
        "cargo": ("Груз: {category}.", "Характер груза — {category}."),
        "load": ("{pieces}, общий вес {weight}.", "Всего {pieces}, вес {weight}.",
                 "Количество: {per_piece}.", "Количество мест и вес: {per_piece}.",
                 "Количество мест: {pieces}.", "Общий вес груза — {weight}."),
        "equipment": ("Требуется {equipment}.", "Нужен {equipment}.",
                      "Тип транспорта: {equipment}."),
        "dates": ("Загрузка {pickup}, доставка {delivery}.",
                  "Дата загрузки — {pickup}, доставка — {delivery}.",
                  "Загрузка {pickup}.", "Дата загрузки — {pickup}.",
                  "Доставка {delivery}.", "Доставить нужно {delivery}."),
        "bye": ("Спасибо!", "Заранее спасибо.", "Ждём ваше предложение.",
                "С уважением, отдел логистики."),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("hello", "shipper", "route", "load", "weight_conflict", "pieces_conflict", "cargo",
         "equipment", "dates", "distractors", "bye"),
        ("hello", "route", "dates", "load", "weight_conflict", "pieces_conflict", "equipment",
         "cargo", "shipper", "distractors", "bye"),
        ("hello", "shipper", "cargo", "load", "weight_conflict", "pieces_conflict", "route",
         "equipment", "dates", "distractors", "bye"),
        ("hello", "route", "equipment", "load", "weight_conflict", "pieces_conflict", "dates",
         "cargo", "distractors", "shipper", "bye"),
        ("hello", "shipper", "dates", "route", "load", "weight_conflict", "pieces_conflict",
         "equipment", "cargo", "distractors", "bye"),
        ("hello", "equipment", "route", "cargo", "load", "weight_conflict", "pieces_conflict",
         "dates", "shipper", "distractors", "bye"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T2")
class MessengerRu(LayoutFamily):
    name = "T2"
    language = "ru"
    slots = {
        "hi": ("Привет!", "Коллеги, привет.", "Добрый!"),
        "route": ("Нужна машина {origin_from} {destination_to}.",
                  "Есть груз {origin_from} {destination_to}.",
                  "{origin_from} {destination_to}, нужна машина.",
                  "Нужна машина {origin_from}.", "Есть груз {destination_to}."),
        "load": ("{pieces}, {weight}.", "Вес {weight}, {pieces}.", "{per_piece}.",
                 "{pieces}.", "Вес {weight}."),
        "equipment": ("Нужен {equipment}.", "Кузов — {equipment}."),
        "dates": ("Грузимся {pickup}, выгрузка {delivery}.",
                  "Загрузка {pickup}, выгрузка {delivery}.",
                  "Загрузка {pickup}.", "Грузимся {pickup}.", "Выгрузка {delivery}."),
        "cargo": ("Груз — {category}.", "Что везём: {category}."),
        "shipper": ("Клиент — {shipper}.", "Это для {shipper}."),
        "end": ("Спасибо.", "Скиньте ставку.", "Жду ставку."),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("hi", "route", "load", "weight_conflict", "pieces_conflict", "equipment", "dates",
         "cargo", "shipper", "distractors", "end"),
        ("route", "dates", "load", "weight_conflict", "pieces_conflict", "equipment", "cargo",
         "distractors", "shipper", "end"),
        ("hi", "equipment", "route", "dates", "load", "weight_conflict", "pieces_conflict",
         "shipper", "cargo", "distractors"),
        ("route", "equipment", "load", "weight_conflict", "pieces_conflict", "cargo", "dates",
         "shipper", "distractors", "end"),
        ("hi", "shipper", "route", "cargo", "load", "weight_conflict", "pieces_conflict",
         "equipment", "dates", "distractors", "end"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T3")
class ListRu(LayoutFamily):
    name = "T3"
    language = "ru"
    joiner = "\n"
    condition_style = "list"
    slots = {
        "header": ("Заявка на перевозку", "Прошу рассчитать перевозку:", "Данные по грузу:"),
        "origin": ("Откуда: {origin}", "Пункт загрузки: {origin}", "Место погрузки: {origin}"),
        "destination": ("Куда: {destination}", "Пункт выгрузки: {destination}",
                        "Место доставки: {destination}"),
        "pickup": ("Дата загрузки: {pickup}", "Загрузка: {pickup}"),
        "delivery": ("Дата доставки: {delivery}", "Доставка: {delivery}"),
        "cargo": ("Груз: {category}", "Наименование груза: {category}"),
        "pieces": ("Количество мест: {pieces}", "Грузовые места: {pieces}",
                   "Места: {per_piece}", "Количество и вес мест: {per_piece}"),
        "weight": ("Вес: {weight}", "Общий вес: {weight}"),
        "equipment": ("Тип кузова: {equipment}", "Транспорт: {equipment}"),
        "shipper": ("Отправитель: {shipper}", "Грузоотправитель: {shipper}"),
        "footer": ("Спасибо.", "Жду ставку."),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("header", "origin", "destination", "pickup", "delivery", "cargo", "pieces", "weight",
         "weight_conflict", "pieces_conflict", "equipment", "shipper", "distractors", "footer"),
        ("header", "shipper", "origin", "destination", "cargo", "weight", "pieces",
         "weight_conflict", "pieces_conflict", "equipment", "pickup", "delivery", "distractors"),
        ("header", "pickup", "origin", "delivery", "destination", "equipment", "cargo", "pieces",
         "weight", "weight_conflict", "pieces_conflict", "shipper", "distractors", "footer"),
        ("origin", "destination", "equipment", "weight", "pieces", "weight_conflict",
         "pieces_conflict", "cargo", "pickup", "delivery", "shipper", "distractors"),
        ("header", "cargo", "pieces", "weight", "weight_conflict", "pieces_conflict", "origin",
         "destination", "pickup", "delivery", "equipment", "distractors", "shipper", "footer"),
    )  # fmt: skip


@TEMPLATE_FAMILIES.register("T7")
class SlangRu(LayoutFamily):
    """Short chat with slang and abbreviations («реф», «палл.», «12,6т»); OOD families only."""

    name = "T7"
    language = "ru"
    ood_only = True
    joiner = "\n"
    compact = True
    capitalize = False
    condition_style = "slang"
    condition_anchor = "lead"
    slots = {
        "lead": ("нужен {equipment}", "ищу {equipment}", "{equipment} нужен"),
        "route": ("{origin} → {destination}", "{origin} - {destination}",
                  "{origin_from} {destination_to}", "{origin_from}", "{destination_to}"),
        "load": ("{weight}, {pieces}", "{pieces}, {weight}", "{per_piece}", "{pieces}",
                 "{weight}"),
        "dates": ("загрузка {pickup}, выгрузка {delivery}", "грузимся {pickup}, сдать {delivery}",
                  "грузимся {pickup}", "погрузка {pickup}", "выгрузка {delivery}"),
        "cargo": ("груз {category}", "везём: {category}"),
        "shipper": ("от {shipper}", "клиент {shipper}"),
        "end": ("срочно", "кто свободен?", "ставку в личку", "есть кто?"),
        **_NOTES,
    }  # fmt: skip
    orders = (
        ("lead", "route", "load", "weight_conflict", "pieces_conflict", "dates", "cargo",
         "shipper", "distractors", "end"),
        ("route", "lead", "dates", "load", "weight_conflict", "pieces_conflict", "shipper",
         "cargo", "distractors", "end"),
        ("lead", "load", "weight_conflict", "pieces_conflict", "route", "cargo", "dates",
         "distractors", "shipper"),
        ("route", "load", "weight_conflict", "pieces_conflict", "lead", "cargo", "shipper",
         "dates", "distractors", "end"),
        ("shipper", "lead", "route", "dates", "cargo", "load", "weight_conflict",
         "pieces_conflict", "distractors", "end"),
    )  # fmt: skip
