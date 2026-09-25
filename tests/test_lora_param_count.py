"""The number of LoRA parameters (plan step 12): arithmetic for Mistral-Nemo without weights,
and the real count on the tiny model."""

from __future__ import annotations

import json
from typing import Any

import pytest

from qf.training import MistralDims, lora_param_count, module_shapes
from tests.conftest import REPO_ROOT

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# config.json of mistralai/Mistral-Nemo-Instruct-2407 at 04d8a90549d2 (the fields that matter)
NEMO_CONFIG: dict[str, Any] = {
    "hidden_size": 5120, "intermediate_size": 14336, "num_hidden_layers": 40,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 131072,
    "tie_word_embeddings": False,
}  # fmt: skip
EXPECTED_NEMO_R16 = 57_016_320  # expected on the GPU at step 13


def test_expected_count_for_nemo_config() -> None:
    dims = MistralDims.from_config(NEMO_CONFIG)
    assert dims is not None
    assert module_shapes(dims) == {
        "q_proj": (5120, 4096), "k_proj": (5120, 1024), "v_proj": (5120, 1024),
        "o_proj": (4096, 5120), "gate_proj": (5120, 14336), "up_proj": (5120, 14336),
        "down_proj": (14336, 5120),
    }  # fmt: skip
    assert lora_param_count(dims, TARGETS, 16) == EXPECTED_NEMO_R16
    assert lora_param_count(dims, ["q_proj", "v_proj"], 8) == 40 * 8 * (9216 + 6144)


def test_downloaded_config_matches() -> None:
    path = (REPO_ROOT / "artifacts/base_model/mistralai--Mistral-Nemo-Instruct-2407"
            / "04d8a90549d23fc6bd7f642064003592df51e9b3" / "config.json")  # fmt: skip
    if not path.exists():
        pytest.skip("run `uv run qf tokens fetch` first")
    assert MistralDims.from_config(json.loads(path.read_text("utf-8"))) == MistralDims.from_config(
        NEMO_CONFIG
    )


def test_unknown_target_rejected() -> None:
    dims = MistralDims.from_config(NEMO_CONFIG)
    assert dims is not None
    with pytest.raises(ValueError, match="unknown target modules"):
        lora_param_count(dims, ["q_proj", "lm_head"], 16)


def _tiny_model() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import MistralConfig, MistralForCausalLM

    from qf.training.trainers.hf_qlora.model_loading import TINY_DIMS

    return MistralForCausalLM(MistralConfig(vocab_size=50, **TINY_DIMS))


def test_lora_count_formula_tiny() -> None:
    from qf.contracts import TrainLora
    from qf.training.trainers.hf_qlora.model_loading import apply_lora, count_trainable

    model = _tiny_model()
    dims = MistralDims.from_config(model.config.to_dict())
    assert dims is not None
    peft_model = apply_lora(model, TrainLora(rank=4, alpha=8, dropout=0.0, target_modules=TARGETS))
    trainable, total = count_trainable(peft_model)
    # layers × r × Σ(in + out): q 64→64, k/v 64→32, o 64→64, gate/up 64→128, down 128→64
    assert (
        trainable
        == lora_param_count(dims, TARGETS, 4)
        == 2 * 4 * (128 + 96 + 96 + 128 + 192 + 192 + 192)
    )
    assert total > trainable


def test_missing_target_module_raises() -> None:
    from qf.common import QFError
    from qf.contracts import TrainLora
    from qf.training.trainers.hf_qlora.model_loading import apply_lora, check_target_modules

    model = _tiny_model()
    assert check_target_modules(model, TARGETS) == 2
    with pytest.raises(QFError, match=r"\['c_attn'\] are not in layer 0; its modules"):
        apply_lora(model, TrainLora(rank=4, alpha=8, dropout=0.0,
                                    target_modules=["q_proj", "c_attn"]))  # fmt: skip
