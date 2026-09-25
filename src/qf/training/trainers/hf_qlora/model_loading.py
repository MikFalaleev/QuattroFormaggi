"""Loading the model to train and attaching LoRA (plan step 12).

The loader decides the device and precision; `train.py` only reads them, so one code path
serves the real QLoRA run and the tiny CPU tests. bitsandbytes is imported only after CUDA is
found, so a machine without a GPU gets "CUDA not available", not an import error.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import MistralConfig, MistralForCausalLM

from qf.common import QFError
from qf.contracts import TrainConfig, TrainLora
from qf.training import BaseModelConfig, verify_base_weights
from qf.training.trainers.hf_qlora.config import HFQLoRATrainerConfig

__all__ = [
    "LOADERS",
    "LoadedModel",
    "apply_lora",
    "check_target_modules",
    "count_trainable",
    "load_base_for_training",
    "load_tiny_for_test",
]

Device = Literal["cuda", "cpu"]
Precision = Literal["bf16", "fp16", "fp32"]
_LAYER = re.compile(r"\.layers\.(\d+)\.")
# initializer_range 0.1, not Mistral's 0.02: the frozen random lm_head of std 0.02 caps the
# logits near ±1.3 after the final norm, so LoRA alone could not fit even a repeated batch
TINY_DIMS: dict[str, Any] = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                             "num_attention_heads": 4, "num_key_value_heads": 2,
                             "head_dim": 16, "initializer_range": 0.1}  # fmt: skip


@dataclass
class LoadedModel:
    model: Any
    device: Device
    precision: Precision
    quantization: str
    base_model: str | None  # the Hub id of the loaded weights; None for a random test model
    hardware: dict[str, Any] = field(default_factory=dict)


def _gpu_facts() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {"gpu_name": torch.cuda.get_device_name(0), "gpu_memory_bytes": int(props.total_memory),
            "gpu_count": torch.cuda.device_count(), "cuda_version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))}  # fmt: skip


def load_base_for_training(cfg: TrainConfig, base: BaseModelConfig, root: Path,
                           params: HFQLoRATrainerConfig, tok: Any) -> LoadedModel:  # fmt: skip
    """The pinned base in nf4 on GPU 0, prepared for k-bit training, from local files only."""
    if not torch.cuda.is_available():
        raise QFError("CUDA not available: QLoRA training needs an NVIDIA GPU; run it on the GPU "
                      "machine (plan step 13). Tests and rehearsals use the tiny_random_cpu "
                      "loader (configs/train/tiny_cpu_test.yaml)")  # fmt: skip
    if not cfg.quantization.load_in_4bit:
        raise QFError("the cuda_4bit loader is QLoRA: quantization.load_in_4bit must be true")
    bf16 = cfg.quantization.compute_dtype == "bf16"
    if bf16 and not torch.cuda.is_bf16_supported():
        raise QFError("this GPU does not support bf16: set quantization.compute_dtype to fp16 "
                      "after checking fp16 stability (plan step 13)")  # fmt: skip
    verify_base_weights(base, root)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True, bnb_4bit_quant_type=cfg.quantization.type,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant, bnb_4bit_compute_dtype=dtype,
    )  # fmt: skip
    directory = base.directory(root)
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=dtype, attn_implementation=params.attn_implementation,
    )  # fmt: skip
    model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )  # fmt: skip
    model.config.use_cache = False
    return LoadedModel(model, "cuda", cfg.quantization.compute_dtype,
                       f"{cfg.quantization.type}, double quant {cfg.quantization.double_quant} "
                       "(bitsandbytes)", base.id, _gpu_facts())  # fmt: skip


def load_tiny_for_test(cfg: TrainConfig, base: BaseModelConfig, root: Path,
                       params: HFQLoRATrainerConfig, tok: Any) -> LoadedModel:  # fmt: skip
    """A randomly initialised 2-layer Mistral sized for the base model's tokenizer, fp32 on CPU.
    Its weights come from the global seed: the caller seeds before loading."""
    settings: dict[str, Any] = {
        **TINY_DIMS, "attn_implementation": params.attn_implementation,
        "max_position_embeddings": max(512, cfg.training.max_sequence_length),
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
    }  # fmt: skip
    config = MistralConfig(vocab_size=len(tok), **settings)
    model = MistralForCausalLM(config)  # type: ignore[no-untyped-call]
    model.config.use_cache = False
    return LoadedModel(model, "cpu", "fp32", "none (tiny random model, not a real model)", None)


LOADERS: dict[str, Callable[[TrainConfig, BaseModelConfig, Path, HFQLoRATrainerConfig, Any],
                            LoadedModel]] = {
    "cuda_4bit": load_base_for_training,
    "tiny_random_cpu": load_tiny_for_test,
}  # fmt: skip


def check_target_modules(model: Any, targets: list[str]) -> int:
    """Every target module must exist in every decoder layer; returns the number of layers."""
    per_layer: dict[int, set[str]] = {}
    for name, _ in model.named_modules():
        match = _LAYER.search(f".{name}.")
        if match:
            per_layer.setdefault(int(match.group(1)), set()).add(name.rsplit(".", 1)[-1])
    if not per_layer:
        raise QFError("no decoder layers (`.layers.<n>.`) found in the model")
    for index, names in sorted(per_layer.items()):
        absent = [t for t in targets if t not in names]
        if absent:
            raise QFError(f"target modules {absent} are not in layer {index}; its modules: "
                          f"{sorted(names)}")  # fmt: skip
    return len(per_layer)


def apply_lora(model: Any, lora: TrainLora) -> PeftModel:
    """LoRA on the target modules. The adapter keeps the local base directory as its base
    while training: with a Hub id, peft would query the Hub on every save (`train.py` writes the
    Hub id and revision into the final adapter_config.json)."""
    check_target_modules(model, lora.target_modules)
    config = LoraConfig(r=lora.rank, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                        target_modules=list(lora.target_modules), bias="none",
                        task_type="CAUSAL_LM")  # fmt: skip
    return get_peft_model(model, config)


def count_trainable(model: Any) -> tuple[int, int]:
    """(trainable, total) parameters; every trainable parameter must be a LoRA weight."""
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    foreign = [n for n, _ in trainable if "lora_" not in n]
    if foreign:
        raise QFError(f"parameters outside LoRA would be trained: {foreign[:5]}")
    _, total = model.get_nb_trainable_parameters()
    return sum(n for _, n in trainable), int(total)
