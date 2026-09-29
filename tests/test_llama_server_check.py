"""Step 17 check with a running llama-server of the pinned llama.cpp (D-123), before the evals.

On the GPU machine, with a GGUF served on 127.0.0.1:8081:
`uv run pytest -m llama_server --run-llama-server -s tests/test_llama_server_check.py`.
The prompt the server builds (its chat template, BOS added once when tokenizing) must equal
the training prompt on 5 bench records; for a merged model (alias `qf-merged-…`) the first 5
val records — answered exactly after training (E302, E304) — must come back equal to gold.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import httpx
import pytest

from qf.domain import parse_sft_jsonl
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.llama_server
URL = os.environ.get("QF_LLAMA_SERVER", "http://127.0.0.1:8081")
BASE_DIR = REPO_ROOT / "artifacts/base_model/mistralai--Mistral-Nemo-Instruct-2407" / (
    "04d8a90549d23fc6bd7f642064003592df51e9b3")  # fmt: skip


def _post(path: str, body: dict[str, Any]) -> Any:
    response = httpx.post(URL + path, json=body, timeout=120)
    response.raise_for_status()
    return response.json()


def _alias() -> str:
    return str(httpx.get(URL + "/v1/models", timeout=30).json()["data"][0]["id"])


def _records(path: str, n: int) -> list[Any]:
    records, _ = parse_sft_jsonl((REPO_ROOT / path).read_text(encoding="utf-8"), source=path)
    return records[:n]


def test_server_prompt_equals_training_prompt() -> None:
    from qf.training import chat_ids, load_tokenizer

    tok = load_tokenizer(BASE_DIR, fix_mistral_regex=True)
    print(f"\n{_alias()}")
    for record in _records("data/splits_v2/bench_v2.jsonl", 5):
        messages = [{"role": m.role, "content": m.content} for m in record.messages[:2]]
        expected = chat_ids(tok, messages, add_generation_prompt=True)
        prompt = _post("/apply-template", {"messages": messages})["prompt"]
        # as the chat endpoint does: tokenize with special tokens added and parsed
        served = _post("/tokenize", {"content": prompt, "add_special": True,
                                     "parse_special": True})["tokens"]  # fmt: skip
        assert served == expected, f"{record.id}: server prompt differs from training"
        assert served.count(tok.bos_token_id) == 1
    print("prompt of the server = training prompt on 5 of 5 records, one BOS")


def test_merged_model_answers_first_val_records_exactly() -> None:
    from qf.backends import OpenAILocalBackend
    from qf.eval.harness import GenerationSettings, _request

    alias = _alias()
    if not alias.startswith("qf-merged"):
        pytest.skip(f"{alias}: not a merged model")
    backend = OpenAILocalBackend(OpenAILocalBackend.Config(base_url=URL + "/v1",
                                                           model_substring=alias))  # fmt: skip
    exact = 0
    for record in _records("data/processed/generated_v2/val.jsonl", 5):
        result = backend.generate(_request(record, GenerationSettings(), None))
        with contextlib.suppress(ValueError):
            exact += json.loads(result.text) == json.loads(record.messages[2].content)
    print(f"\n{alias}: first 5 val records equal to gold: {exact} of 5")
    assert exact >= 3, "the fine-tuning does not seem to be in this GGUF"
