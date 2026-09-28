"""GPU check of `hf_local` with the real model before the evaluation of step 14 (D-120).

On the GPU machine, after `qf train fetch-base` and with the E302 adapter in `runs/`:
`uv run pytest -m gpu --run-gpu -s tests/test_gpu_hf_local.py`. It prints whether a batch
answers exactly as single requests (a mismatch in 4-bit bf16 is a numerics finding, recorded,
not a failure: base and adapter runs share one batch size) and the throughput per batch size.
It fails if the adapter does not take effect: peft only warns when weights do not load, and the
"adapter" would silently answer as the base (the first val records, which E302 answered exactly).
"""

from __future__ import annotations

import json
import os
import time
from functools import cache
from typing import Any

import pytest

from qf.common import load_yaml_config
from qf.domain import parse_sft_jsonl
from qf.eval import EvalConfig, load_bench
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.gpu
CONFIG = REPO_ROOT / "configs/eval/hf_adapter_s42_bench_v2.yaml"
N_SINGLE = 6


@cache
def setup() -> tuple[Any, list[Any]]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    os.environ.setdefault("QF_PROJECT_ROOT", str(REPO_ROOT))
    from qf.backends.hf_local import HFLocalBackend, HFLocalConfig
    from qf.eval.harness import _request

    cfg = load_yaml_config(CONFIG, EvalConfig)
    _, records = load_bench(cfg, REPO_ROOT)
    impl = HFLocalBackend(HFLocalConfig(**(cfg.backend.model_extra or {})))
    return impl, [_request(r, cfg.generation, None) for r in records]


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def test_gpu_adapter_takes_effect() -> None:
    impl, _ = setup()
    lora_b = [p for n, p in impl._model.named_parameters() if "lora_B" in n]
    assert lora_b and any(bool(p.abs().sum() > 0) for p in lora_b), "LoRA weights are zero"
    from qf.eval.harness import GenerationSettings, _request

    text = (REPO_ROOT / "data/processed/generated_v2/val.jsonl").read_text(encoding="utf-8")
    records, _ = parse_sft_jsonl(text, source="val.jsonl")
    first = records[:5]  # the records of E302's samples.md: exact in 5 of 5 after training
    results = impl.generate_batch([_request(r, GenerationSettings(), None) for r in first])
    pairs = zip(results, first, strict=True)
    exact = sum(_same_json(res.text, r.messages[2].content) for res, r in pairs)
    print(f"\nadapter: {len(lora_b)} lora_B tensors; first 5 val records equal gold: {exact} of 5")
    assert exact >= 3, "the adapter does not seem to be applied"


def _same_json(a: str, b: str) -> bool:
    try:
        return bool(json.loads(a) == json.loads(b))
    except ValueError:
        return False


def test_gpu_batch_answers_as_single_requests() -> None:
    impl, reqs = setup()
    print(f"\n{impl.model_id()}")
    single = [impl.generate(r) for r in reqs[:N_SINGLE]]
    batched = impl.generate_batch(reqs[:N_SINGLE])
    same = sum(a.text == b.text for a, b in zip(single, batched, strict=True))
    print(f"batch of {N_SINGLE} equals single requests: {same} of {N_SINGLE}; "
          f"JSON parses: single {sum(_parses(r.text) for r in single)}, "
          f"batch {sum(_parses(r.text) for r in batched)}")  # fmt: skip
    assert all(r.ok for r in single + batched)


@pytest.mark.parametrize("size", [8, 16, 32])
def test_gpu_throughput_per_batch_size(size: int) -> None:
    impl, reqs = setup()
    started = time.monotonic()
    results = impl.generate_batch(reqs[:size])
    seconds = time.monotonic() - started
    tokens = sum(r.completion_tokens or 0 for r in results)
    print(f"\nbatch {size}: {seconds:.1f} s, {tokens} new tokens, {tokens / seconds:.0f} tok/s, "
          f"{seconds / size:.2f} s per case; all ok: {all(r.ok for r in results)}")  # fmt: skip
    assert all(r.ok for r in results)
