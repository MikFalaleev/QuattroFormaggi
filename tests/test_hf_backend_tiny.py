"""The `hf_local` backend on a tiny random model on the CPU (plan step 14, D-120): the prompt of
training, only new tokens, greedy and deterministic, a batch answers as single requests, errors
are results, adapters are checked against their manifest."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from qf.backends.hf_local import HFLocalBackend, HFLocalConfig, _prompt_ids  # noqa: E402
from qf.common import QFError  # noqa: E402
from qf.contracts import BatchGenerationBackend, GenerationBackend, Message  # noqa: E402
from qf.training import chat_ids  # noqa: E402
from tests.tiny_hf import (  # noqa: E402
    ADAPTER,
    TINY_BASE,
    hf_config,
    tiny_adapter,
    tiny_base,
    tiny_request,
)
from tests.tiny_training import REVISION  # noqa: E402


@pytest.fixture
def root(fake_project: Path) -> Path:
    tiny_base(fake_project)
    return fake_project


def backend(**changes: Any) -> HFLocalBackend:
    return HFLocalBackend(HFLocalConfig(**hf_config(**changes)))


def test_hf_backend_is_a_batch_backend(root: Path) -> None:
    impl = backend()
    assert isinstance(impl, GenerationBackend) and isinstance(impl, BatchGenerationBackend)
    assert impl.name == "hf_local" and impl.batch_size == 4
    assert impl.model_id() == f"fake/tiny@{REVISION[:12]} fp32"


def test_prompt_ids_are_those_of_training(root: Path) -> None:
    impl = backend()
    req = tiny_request(3)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    assert _prompt_ids(impl._tok, req) == chat_ids(impl._tok, messages, add_generation_prompt=True)


def test_hf_backend_returns_only_new_tokens(root: Path) -> None:
    impl = backend()
    req = tiny_request(1)
    result = impl.generate(req)
    prompt = _prompt_ids(impl._tok, req)
    model = AutoModelForCausalLM.from_pretrained(root / TINY_BASE)
    ids = torch.tensor([prompt])
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                         max_new_tokens=req.max_tokens, pad_token_id=impl._tok.pad_token_id,
                         eos_token_id=impl._tok.eos_token_id)  # fmt: skip
    expected = impl._tok.decode(out[0, len(prompt) :], skip_special_tokens=True).strip()
    assert result.ok and result.text == expected
    assert result.prompt_tokens == len(prompt) and 0 < (result.completion_tokens or 0) <= 12


def test_greedy_deterministic(root: Path) -> None:
    impl = backend()
    assert impl.generate(tiny_request(2)).text == impl.generate(tiny_request(2)).text


def test_batch_equals_single(root: Path) -> None:
    impl = backend()
    reqs = [tiny_request(i, repeat=r) for i, r in ((0, 1), (7, 3), (13, 2), (22, 5), (31, 1))]
    assert len({len(_prompt_ids(impl._tok, r)) for r in reqs}) > 1
    single = [impl.generate(r).text for r in reqs]
    assert [r.text for r in impl.generate_batch(reqs)] == single


def test_pad_is_the_existing_pad_token(root: Path) -> None:
    impl = backend()
    assert impl._tok.pad_token == "<pad>" and impl._tok.padding_side == "left"
    assert impl._tok.pad_token_id == impl._tok.convert_tokens_to_ids("<pad>")


def test_refusals_are_results_not_exceptions(root: Path) -> None:
    impl = backend()
    ok = tiny_request(0)
    schema = tiny_request(1, json_schema={"type": "object"})
    warm = ok.model_copy(update={"temperature": 0.7})
    answer = Message(role="assistant", content="{}")
    leaked = ok.model_copy(update={"messages": [*ok.messages, answer]})
    results = impl.generate_batch([ok, schema, warm, leaked])
    assert results[0].ok
    assert "json_schema" in (results[1].error or "") and results[1].text == ""
    assert "greedily" in (results[2].error or "")
    assert "must end with a user message" in (results[3].error or "")


def test_a_failed_generation_fails_each_request_of_the_batch(root: Path) -> None:
    impl = backend()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    impl._model.generate = broken
    results = impl.generate_batch([tiny_request(0), tiny_request(1)])
    assert [r.error for r in results] == ["RuntimeError: CUDA out of memory"] * 2


def test_4bit_needs_cuda(root: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(QFError, match="4-bit loading needs CUDA"):
        backend(load_in_4bit=True, device="cuda")


def test_adapter_changes_answers_and_names_itself(root: Path) -> None:
    tiny_adapter(root)
    base = backend()
    tuned = backend(adapter=ADAPTER.as_posix(), allow_test_adapter=True)
    assert tuned.model_id().startswith(f"fake/tiny@{REVISION[:12]} fp32 + lora ")
    reqs = [tiny_request(i, max_tokens=8) for i in range(4)]
    tuned_texts = [r.text for r in tuned.generate_batch(reqs)]
    assert tuned_texts != [r.text for r in base.generate_batch(reqs)]


def test_adapter_of_the_cpu_rehearsal_is_refused(root: Path) -> None:
    tiny_adapter(root)
    with pytest.raises(QFError, match="loader 'tiny_random_cpu'"):
        backend(adapter=ADAPTER.as_posix())


OTHER_BASES = [("revision", "b" * 40), ("base_model", "other/model"),
               ("quantization", "nf4, double quant True (bitsandbytes)")]  # fmt: skip


@pytest.mark.parametrize(("field", "value"), OTHER_BASES)
def test_adapter_of_another_base_is_refused(root: Path, field: str, value: str) -> None:
    tiny_adapter(root, **{field: value})
    with pytest.raises(QFError, match=f"adapter refused: {field}"):
        backend(adapter=ADAPTER.as_posix(), allow_test_adapter=True)


def test_adapter_without_its_artifact_manifest_is_refused(root: Path) -> None:
    tiny_adapter(root)
    (root / ADAPTER).with_name("adapter.manifest.json").unlink()
    with pytest.raises(QFError):
        backend(adapter=ADAPTER.as_posix(), allow_test_adapter=True)


@pytest.mark.needs_tokenizer
def test_real_tokenizer_prompts_equal_training(fake_project: Path) -> None:
    """The pinned Mistral-Nemo tokenizer: the backend's prompt ids of bench_v2 records are
    exactly training's, and the pad is the `<pad>` of the vocabulary (id 10, D-113)."""
    from qf.domain import parse_sft_jsonl
    from qf.training import choose_pad_token, load_tokenizer
    from tests.conftest import REPO_ROOT

    base = "mistralai--Mistral-Nemo-Instruct-2407/04d8a90549d23fc6bd7f642064003592df51e9b3"
    directory = REPO_ROOT / "artifacts/base_model" / base
    if not (directory / "tokenizer.json").exists():
        pytest.skip("the pinned tokenizer is not fetched (qf tokens fetch)")
    shell = SimpleNamespace(cfg=HFLocalConfig(**hf_config(fix_mistral_regex=True)))
    ours = HFLocalBackend._load_tokenizer(shell, directory)  # type: ignore[arg-type]
    theirs = load_tokenizer(directory, fix_mistral_regex=True)
    assert ours.pad_token_id == choose_pad_token(theirs)[0] == 10
    text = (REPO_ROOT / "data/splits_v2/bench_v2.jsonl").read_text(encoding="utf-8")
    records, _ = parse_sft_jsonl(text, source="bench_v2.jsonl")
    for record in records[:: max(1, len(records) // 20)]:
        messages = [{"role": m.role, "content": m.content} for m in record.messages[:2]]
        req = tiny_request(0).model_copy(update={"messages": record.messages[:2]})
        assert _prompt_ids(ours, req) == chat_ids(theirs, messages, add_generation_prompt=True)
    assert json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
