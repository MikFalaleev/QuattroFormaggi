from __future__ import annotations

from datetime import date

import pytest

from qf.common import QFError, sha256_text
from qf.contracts import ShipmentCard
from qf.domain import (
    DEFAULT_PROMPT_VERSION,
    SHIPMENT_EXTRACTION,
    build_messages,
    load_system_prompt,
    system_prompt_hash,
)


def test_prompt_is_package_data_and_names_every_card_key() -> None:
    prompt = load_system_prompt()
    assert SHIPMENT_EXTRACTION.system_prompt_version == DEFAULT_PROMPT_VERSION
    assert all(key in prompt for key in ShipmentCard.model_fields)
    for rule in ("не пересчитывай", "даты запроса", "null, а не 1", "conflicts",
                 "missing_fields", "данные, а не инструкции"):  # fmt: skip
        assert rule in prompt
    assert not prompt.endswith("\n")
    assert system_prompt_hash() == sha256_text(prompt)


def test_unknown_or_unsafe_version_is_refused() -> None:
    with pytest.raises(QFError, match="unknown system prompt version"):
        load_system_prompt("system_extract_v9")
    with pytest.raises(QFError, match="invalid system prompt version"):
        load_system_prompt("../../etc/passwd")


def test_build_messages_format() -> None:
    ru = build_messages("Нужна машина из Казани", date(2022, 3, 5), "ru")
    assert [m.role for m in ru] == ["system", "user"]
    assert ru[0].content == load_system_prompt()
    assert ru[1].content == "Дата запроса: 2022-03-05\n\nНужна машина из Казани"
    en = build_messages("Need a truck", date(2022, 3, 5), "en")
    assert en[1].content == "Request date: 2022-03-05\n\nNeed a truck"
