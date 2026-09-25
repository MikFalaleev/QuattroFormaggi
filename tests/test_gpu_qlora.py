"""GPU checks of the real QLoRA setup (plan step 13), prepared at step 12.

Run only on the GPU machine after `qf train fetch-base` (⛔ STOP: ~24.5 GB):
`uv run pytest -m gpu --run-gpu -s tests/test_gpu_qlora.py`. The model is loaded once.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import pytest

from qf.training import base_model_for, load_records, load_train_config
from tests.conftest import REPO_ROOT
from tests.test_lora_param_count import EXPECTED_NEMO_R16

pytestmark = pytest.mark.gpu
CONFIG = REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml"


@cache
def peft_model() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from qf.training.trainers.hf_qlora.config import HFQLoRATrainerConfig
    from qf.training.trainers.hf_qlora.model_loading import LOADERS, apply_lora

    cfg = load_train_config(CONFIG)
    loaded = LOADERS["cuda_4bit"](cfg, base_model_for(cfg, REPO_ROOT), REPO_ROOT,
                                  HFQLoRATrainerConfig(loader="cuda_4bit"))  # fmt: skip
    return apply_lora(loaded.model, cfg.lora), loaded.tokenizer


def test_gpu_bf16_decision() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    supported = torch.cuda.is_bf16_supported()
    chosen = load_train_config(CONFIG).quantization.compute_dtype
    print(f"\n{torch.cuda.get_device_name(0)}: bf16 supported={supported}; config: {chosen}")
    assert chosen == ("bf16" if supported else "fp16"), "set quantization.compute_dtype explicitly"


def test_gpu_target_modules_present() -> None:
    from qf.training.trainers.hf_qlora.model_loading import check_target_modules

    model, _ = peft_model()
    assert check_target_modules(model, load_train_config(CONFIG).lora.target_modules) == 40


def test_gpu_trainable_params_equals_57016320() -> None:
    from qf.training.trainers.hf_qlora.model_loading import count_trainable

    model, _ = peft_model()
    trainable, total = count_trainable(model)
    print(f"\ntrainable {trainable:,} of {total:,}")
    assert trainable == EXPECTED_NEMO_R16


def test_gpu_one_forward_backward_no_nan() -> None:
    import torch

    from qf.training import build_features

    model, tok = peft_model()
    cfg = load_train_config(CONFIG)
    record = load_records(REPO_ROOT / cfg.data.train_path, cfg.data.expected_sha256.train)[0]
    features = build_features(record, tok, cfg.training.max_sequence_length)
    batch = {key: torch.tensor([value]).to("cuda") for key, value in features.as_dict().items()}  # type: ignore[union-attr]
    model.train()
    torch.cuda.reset_peak_memory_stats()
    loss = model(**batch).loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    print(f"\nloss {loss.item():.4f}, peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} "
          f"GiB, {len(grads)} LoRA gradients")  # fmt: skip
    assert math.isfinite(loss.item()) and grads
    assert all(torch.isfinite(g).all() for g in grads)
    model.zero_grad()
