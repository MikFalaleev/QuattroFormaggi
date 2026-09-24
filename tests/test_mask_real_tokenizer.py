"""The loss mask on the real Mistral-Nemo tokenizer (plan step 11, marker `needs_tokenizer`).

Needs the pinned tokenizer files (`uv run qf tokens fetch`) and generated_v2; run with
`uv run pytest -m needs_tokenizer`. A failure here is a stop for a human, never a reason to
weaken the test (plan step 11).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pytest

from qf.common import load_yaml_config
from qf.contracts import SFTRecord
from qf.domain import parse_sft_jsonl
from qf.training import (
    IGNORE_INDEX,
    BaseModelConfig,
    Features,
    Skip,
    build_features,
    chat_ids,
    check_template,
    load_tokenizer,
)
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.needs_tokenizer
CONFIG = load_yaml_config(REPO_ROOT / "configs/train/base_model.yaml", BaseModelConfig)
DATA = REPO_ROOT / "data/processed/generated_v2"
MAX_LEN = 2048


@cache
def tokenizer() -> Any:
    directory = CONFIG.directory(REPO_ROOT)
    if not (directory / "tokenizer.json").exists():
        pytest.skip("run `uv run qf tokens fetch` first")
    pytest.importorskip("transformers")
    return load_tokenizer(directory, fix_mistral_regex=CONFIG.fix_mistral_regex)


@cache
def records() -> tuple[SFTRecord, ...]:
    """smoke + the first 200 train records (plan step 11)."""
    if not (DATA / "train.jsonl").exists():
        pytest.skip("run `qf data build --config configs/data/generate_v2.yaml` first")
    loaded: list[SFTRecord] = []
    for split, limit in (("smoke", None), ("train", 200)):
        text = (DATA / f"{split}.jsonl").read_text(encoding="utf-8")
        found, issues = parse_sft_jsonl(text, source=split, split=None)
        assert issues == []
        loaded += found[:limit]
    return tuple(loaded)


def features() -> list[tuple[SFTRecord, Features | Skip]]:
    return [(r, build_features(r, tokenizer(), MAX_LEN)) for r in records()]


def test_real_build_features_all_records() -> None:
    built = features()
    assert len(built) == 250 and all(isinstance(f, Features) for _, f in built)


def test_real_trainable_decodes_to_assistant_json() -> None:
    tok = tokenizer()
    for record, built in features():
        assert isinstance(built, Features)
        trainable = [t for t in built.labels if t != IGNORE_INDEX]
        assert trainable[-1] == tok.eos_token_id
        assert tok.decode(trainable[:-1]).strip() == record.messages[2].content, record.id


def test_real_answer_tokenization_matches_template() -> None:
    for record in records():
        check = check_template(record, tokenizer())
        assert check.answer_tokens_match, record.id
        assert check.system_in_input and not check.system_in_full  # the Mistral template


def test_single_bos() -> None:
    tok = tokenizer()
    for _, built in features():
        assert isinstance(built, Features)
        assert (
            built.input_ids[0] == tok.bos_token_id and built.input_ids.count(tok.bos_token_id) == 1
        )


def test_system_prompt_present_in_full_input_ids() -> None:
    tok = tokenizer()
    for record, built in features():
        assert isinstance(built, Features)
        decoded = tok.decode(built.input_ids)
        assert record.messages[0].content in decoded and record.messages[1].content in decoded


def test_prompt_is_what_the_model_sees_when_used() -> None:
    """The prompt part equals `apply_chat_template(messages[:2], add_generation_prompt=True)`,
    the request LM Studio sends (its prompt token counts matched on bench_v2, D-113)."""
    tok = tokenizer()
    for record, built in features()[:20]:
        assert isinstance(built, Features)
        messages = [{"role": m.role, "content": m.content} for m in record.messages[:2]]
        assert built.input_ids[: built.prompt_tokens] == chat_ids(
            tok, messages, add_generation_prompt=True
        )


def test_too_long_rate_below_1pct() -> None:
    built = features()
    assert sum(isinstance(f, Skip) for _, f in built) / len(built) < 0.01


def test_vocabulary_is_not_extended() -> None:
    tok = tokenizer()
    assert len(tok) == 131072 and tok.convert_tokens_to_ids("<pad>") == 10


def test_config_is_pinned() -> None:
    assert CONFIG.revision == "04d8a90549d23fc6bd7f642064003592df51e9b3"
    assert Path(CONFIG.local_dir).parts[0] == "artifacts"  # outside Git
