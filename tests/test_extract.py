from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from qf.backends import FakeBackend
from qf.cli.main import main
from qf.common import project_root, sha256_text
from qf.contracts import GenerationRequest, GenerationResult
from qf.data import generate_records
from qf.domain import build_messages, serialize_target, system_prompt_hash
from qf.runtime import (
    ASSUMPTION_DEFAULT_PIECES,
    ExtractionResult,
    ExtractSettings,
    detect_language,
    extract_card,
)
from tests.factories import make_card_v2, make_target_v2
from tests.generation import synthetic_facts_v2
from tests.test_generate_v2 import small_v2

TEXT = "Нужна рефрижераторная фура Пермь - Казань, 18 паллет, 12,6 т, +2..+6"
REQUEST_DATE = date(2024, 3, 10)
NOW = datetime(2024, 3, 10, 12, 0, tzinfo=UTC)


def _backend(answer: str) -> FakeBackend:
    user = build_messages(TEXT, REQUEST_DATE, "ru", "system_extract_v2")[1].content
    return FakeBackend.from_answers({user: answer})


def _run(answer: str, **kwargs: Any) -> ExtractionResult:
    return extract_card(TEXT, REQUEST_DATE, "ru", _backend(answer), now=NOW, **kwargs)


def test_extract_valid_output() -> None:
    result = _run(serialize_target(make_target_v2()))
    assert result.status == "ok"
    assert result.card is not None and result.card["pieces"] == 18
    assert result.missing_fields == [] and result.warnings == []
    assert result.review_required is False
    assert result.derived.weight_total_kg == 12600.0
    assert result.derived.pieces == 18
    assert result.schema_version == "card_v2"
    assert result.prompt_hash == system_prompt_hash("system_extract_v2")
    assert result.checked_at == "2024-03-10T12:00:00+00:00"
    assert result.model_id.startswith("fake:")


def test_invalid_json_sets_review_required() -> None:
    result = _run("вот ваша карточка: {oops")
    assert result.status == "invalid_output"
    assert result.review_required is True and result.card is None
    assert result.raw_text == "вот ваша карточка: {oops"
    assert result.warnings == ["invalid_output:format"]


def test_schema_violation_is_invalid_output() -> None:
    result = _run(json.dumps({"card": {}, "missing_fields": [], "conflicts": []}))
    assert result.status == "invalid_output"
    assert result.warnings == ["invalid_output:schema"]


def test_backend_error_is_reported() -> None:
    backend = FakeBackend.from_answers({})  # no answer for this request
    result = extract_card(TEXT, REQUEST_DATE, "ru", backend, now=NOW)
    assert result.status == "backend_error" and result.review_required
    assert result.error


def test_missing_mismatch_uses_code_rule() -> None:
    wrong = make_target_v2(make_card_v2(pieces=None), missing_fields=[])  # rule: ["pieces"]
    result = _run(serialize_target(wrong))
    assert result.status == "ok"
    assert result.missing_fields == ["pieces"]
    assert "missing_mismatch" in result.warnings
    assert result.review_required is True


def test_unstated_pieces_assume_one_and_say_so() -> None:
    result = _run(serialize_target(make_target_v2(make_card_v2(pieces=None))))
    assert result.derived.pieces == 1
    assert result.assumptions == [ASSUMPTION_DEFAULT_PIECES]
    assert result.missing_fields == ["pieces"]
    assert result.warnings == []  # a question, not a doubt


def test_unknown_city_warning() -> None:
    from qf.contracts import Place

    card = make_card_v2(origin=Place(city="Атлантида", region="Море"))
    result = _run(serialize_target(make_target_v2(card)))
    assert result.warnings == ["unknown_city:origin"]
    assert result.review_required is True


def test_date_order_warning() -> None:
    card = make_card_v2(pickup_date=date(2024, 3, 12), delivery_date=date(2024, 3, 11))
    result = _run(serialize_target(make_target_v2(card)))
    assert result.warnings == ["delivery_before_pickup"]


def test_pickup_before_request_warning() -> None:
    card = make_card_v2(pickup_date=date(2024, 3, 1), delivery_date=None)
    result = _run(serialize_target(make_target_v2(card)))
    assert result.warnings == ["pickup_before_request"]


def test_request_carries_settings_and_schema_only_when_asked() -> None:
    backend = _backend(serialize_target(make_target_v2()))
    extract_card(TEXT, REQUEST_DATE, "ru", backend, now=NOW)
    assert backend.requests[0].json_schema is None
    assert backend.requests[0].temperature == 0.0
    extract_card(TEXT, REQUEST_DATE, "ru", backend, now=NOW,
            settings=ExtractSettings(json_schema=True, max_tokens=100))  # fmt: skip
    assert backend.requests[1].json_schema is not None
    assert backend.requests[1].max_tokens == 100


def test_messages_identical_to_training_format() -> None:
    records, _ = generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")
    record = records["train"][0]
    seen: list[GenerationRequest] = []

    class Spy:
        name = "spy"

        def model_id(self) -> str:
            return "spy"

        def generate(self, req: GenerationRequest) -> GenerationResult:
            seen.append(req)
            return GenerationResult(text="", latency_s=0.0)

    user = record.messages[1].content
    text = user.split("\n\n", 1)[1]
    request_date = date.fromisoformat(user.split("\n\n", 1)[0].split(": ")[1])
    extract_card(text, request_date, record.language, Spy(), schema_version=record.schema_version)
    assert seen[0].messages == record.messages[:2]


def test_detect_language() -> None:
    assert detect_language("Груз из Перми") == "ru"
    assert detect_language("Load from Perm") == "en"


def test_no_external_fallback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "extract.yaml"
    config.write_text(
        "backend:\n  name: openai_local\n  base_url: http://example.com/v1\n", encoding="utf-8"
    )
    assert main(["extract", "--text", TEXT, "--config", str(config)]) == 1
    assert "non-local endpoint refused" in capsys.readouterr().err


def test_cli_rejects_empty_text(tmp_path: Path) -> None:
    assert main(["extract", "--text", "   "]) == 1


def test_default_config_is_valid() -> None:
    from qf.common import load_yaml_config
    from qf.runtime import ExtractConfig

    cfg = load_yaml_config(project_root() / "configs/runtime/extract.yaml", ExtractConfig)
    assert cfg.schema_version == "card_v2" and cfg.settings.temperature == 0.0


def _fake_config(tmp_path: Path, answer: str) -> Path:
    user = build_messages(TEXT, REQUEST_DATE, "ru", "system_extract_v2")[1].content
    config = tmp_path / "extract.yaml"
    config.write_text(
        json.dumps({"backend": {"name": "fake", "responses": {sha256_text(user): answer}}}),
        encoding="utf-8",
    )
    return config


def test_cli_prints_the_checked_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _fake_config(tmp_path, serialize_target(make_target_v2(make_card_v2(pieces=None))))
    args = ["extract", "--text", TEXT, "--request-date", "2024-03-10", "--config", str(config)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Статус: ok" in out and "Не хватает: pieces" in out
    assert ASSUMPTION_DEFAULT_PIECES in out and "Нужна проверка человеком: нет" in out


def test_cli_json_and_exit_code_of_invalid_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _fake_config(tmp_path, "не JSON")
    request = tmp_path / "request.txt"
    request.write_text(TEXT, encoding="utf-8")
    args = ["extract", "--file", str(request), "--request-date", "2024-03-10",
            "--config", str(config), "--json"]  # fmt: skip
    assert main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid_output" and result["review_required"] is True


@pytest.mark.lmstudio
def test_extract_on_live_lmstudio() -> None:
    """Step 18: `qf extract` against the release model loaded in LM Studio (Q6_K, ctx 4096)."""
    from qf.backends import OpenAILocalBackend

    live = OpenAILocalBackend(OpenAILocalBackend.Config(
        model_substring="quattro-formaggi", expected_context=4096))  # fmt: skip
    result = extract_card(TEXT, REQUEST_DATE, "ru", live)
    assert result.status == "ok", result.error
    assert result.card is not None and result.card["equipment_type"] == "reefer"
    assert result.card["pieces"] == 18 and result.derived.weight_total_kg == 12600.0
    assert result.missing_fields == ["pickup_date"]  # the text names no loading date
    assert result.warnings == []
