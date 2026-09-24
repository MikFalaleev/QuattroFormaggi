"""The lower-bound baselines (D-109): the empty card and the rule-based extractor."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from qf.baselines import EmptyBaseline, RulesBaseline, answer_text, empty_card, extract_card
from qf.baselines.answers import request_date
from qf.common import QFError
from qf.contracts import GenerationRequest, Message
from qf.domain import get_target_schema
from qf.eval import score_prediction_v2
from tests.conftest import REPO_ROOT

SCHEMA = get_target_schema("card_v2")


def ask(backend: Any, text: str) -> str:
    result = backend.generate(GenerationRequest(messages=[Message(role="system", content="s"),
                                                          Message(role="user", content=text)],
                                                max_tokens=8, temperature=0.0))  # fmt: skip
    assert result.ok
    answer: str = result.text
    return answer


def test_empty_baseline() -> None:
    backend = EmptyBaseline(EmptyBaseline.Config())
    target = SCHEMA.parse(ask(backend, "что угодно"))
    assert all(v in (None, []) for v in target.card.model_dump().values())
    assert target.missing_fields == ["origin", "destination", "pickup_date", "equipment_type",
                                     "pieces", "weight_total"]  # fmt: skip
    assert backend.model_id() == "baseline:empty@card_v2"
    v1 = EmptyBaseline(EmptyBaseline.Config(schema_version="card_v1"))
    assert get_target_schema("card_v1").parse(ask(v1, "x")).card.temperature_c is None


def test_rules_extract_a_russian_request() -> None:
    text = (
        "Дата запроса: 2026-09-22\n\nЗдравствуйте! Нужен реф из Казани в Москву, 18 паллет, "
        "общий вес 12,5 т. Держать +2…+6 °C, в машине термописец. Загрузка завтра, "
        "выгрузка 25.09.2026. Грузоотправитель: ООО «Север». Груз — продукты питания."
    )
    card = extract_card(text, request_date(text))
    assert card["origin"] == {"city": "Казань", "region": "Республика Татарстан"}
    assert card["destination"]["city"] == "Москва"
    assert (card["equipment_type"], card["cargo_category"]) == ("reefer", "food_beverage")
    assert (card["pieces"], card["weight_total"]) == (18, {"value": 12.5, "unit": "t"})
    assert (card["pickup_date"], card["delivery_date"]) == ("2026-09-23", "2026-09-25")
    assert card["shipper_name"] == "ООО «Север»"
    assert {"kind": "temperature", "min_c": 2, "max_c": 6} in card["special_conditions"]
    assert {"kind": "sensors", "parameters": ["temperature"]} in card["special_conditions"]


def test_rules_extract_an_english_request() -> None:
    text = (
        "Request date: 2023-04-07\n\nShipper: First Retail\nFrom: Omsk, Omsk Oblast\n"
        "Destination: Perm\nGross weight: 14,405 kg\nNumber of pieces: 10 pallets, 500 kg "
        "each\nTrailer type: lowboy\nDimensions: length 820 cm, width 290 cm\n"
        "Securing: chains and anti-slip mats\nPickup date: April 8, 2023"
    )
    card = extract_card(text, request_date(text))
    assert (card["origin"]["city"], card["destination"]["city"]) == ("Омск", "Пермь")
    assert card["weight_total"] == {"value": 14405, "unit": "kg"} and card["pieces"] == 10
    assert card["weight_per_piece"] is None  # a total is stated: the per-piece weight is dropped
    assert card["equipment_type"] == "lowbed" and card["shipper_name"] == "First Retail"
    assert card["pickup_date"] == "2023-04-08" and card["delivery_date"] is None
    kinds = {c["kind"]: c for c in card["special_conditions"]}
    assert kinds["securing"]["methods"] == ["chains", "anti_slip_mats"]
    assert kinds["oversize"]["length"] == {"value": 820, "unit": "cm"}
    assert kinds["oversize"]["height"] is None


@pytest.mark.parametrize(("text", "expected"), [
    ("Нужен изотерм. Температура не выше +5 °C.", {"min_c": None, "max_c": 5}),
    ("Keep not below +2 °C please", {"min_c": 2, "max_c": None}),
    ("Режим -18 °C.", {"min_c": -18, "max_c": -18}),
    ("Нужен температурный режим, цифры позже.", {"min_c": None, "max_c": None}),
])  # fmt: skip
def test_rules_temperature(text: str, expected: dict[str, Any]) -> None:
    card = extract_card(text, None)
    temperature = next(c for c in card["special_conditions"] if c["kind"] == "temperature")
    assert {k: temperature[k] for k in expected} == expected


def test_rules_never_break_the_answer() -> None:
    backend = RulesBaseline(RulesBaseline.Config())
    assert backend.model_id() == "baseline:rules_v1@card_v2"
    for text in ("", "Дата запроса: 2026-02-30\n\n30 февраля", "+5…-5 °C", "Режим +9…+2 °C"):
        SCHEMA.parse(ask(backend, text))  # always a valid card_v2 answer
    card = extract_card("Режим от +9 до +2 °C", None)
    assert card["special_conditions"][0]["min_c"] == 2  # the range is put in order
    leaked = backend.generate(GenerationRequest(messages=[Message(role="user", content="x"),
                                                          Message(role="assistant", content="y")],
                                                max_tokens=8, temperature=0.0))  # fmt: skip
    assert leaked.error and "not user" in leaked.error
    with pytest.raises(QFError, match="card_v2 only"):
        RulesBaseline(RulesBaseline.Config(schema_version="card_v1"))


def test_invalid_cards_fall_back() -> None:
    bad = {**empty_card("card_v2"), "pieces": 3,
           "special_conditions": [{"kind": "temperature", "min_c": 9, "max_c": 2}]}  # fmt: skip
    target = SCHEMA.parse(answer_text("card_v2", bad))
    assert target.card.pieces == 3 and target.card.special_conditions == []
    worse = {**bad, "pieces": -1}
    assert SCHEMA.parse(answer_text("card_v2", worse)).card.pieces is None


def test_the_rules_supporting_facts() -> None:
    assert request_date("Request date: 2023-04-07\n\ntext") == date(2023, 4, 7)
    assert request_date("no date") is None


@pytest.mark.slow
@pytest.mark.skipif(
    not (REPO_ROOT / "data/processed/generated_v2/val.jsonl").exists(),
    reason="run `qf data build --config configs/data/generate_v2.yaml` first",
)
def test_rules_on_the_validation_split() -> None:
    """The rules were written on generated_v2/val (never on a benchmark); they stay strong on it
    (a regression guard, not a result: results come from bench_v2 runs, E004)."""
    from qf.data import read_sft_records

    records, _ = read_sft_records(REPO_ROOT / "data/processed/generated_v2/val.jsonl")
    backend = RulesBaseline(RulesBaseline.Config())
    keys = []
    for record in records:
        score = score_prediction_v2(record.id, SCHEMA.parse(record.messages[2].content),
                                    ask(backend, record.messages[1].content))  # fmt: skip
        if score.key_fields_correct is not None:
            keys.append(score.key_fields_correct)
    assert sum(keys) / len(keys) > 0.85


def test_package_path() -> None:
    assert (REPO_ROOT / "src/qf/baselines/rules.py").exists() and Path(__file__).exists()
