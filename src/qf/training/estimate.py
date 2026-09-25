"""Training budget arithmetic (plan step 12): steps, processed tokens, GPU hours, cost, memory.

Pure arithmetic: no torch, no GPU. The throughput (tokens/s) is measured on the GPU at step 13;
until then GPU hours and cost are None. Memory figures are rough and checked by measurement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from qf.contracts import Estimate, MemoryEstimate, TokenStats, TrainConfig

__all__ = [
    "LORA_BYTES_PER_PARAM",
    "PLAN_BASE_PARAMS",
    "MistralDims",
    "base_weight_bytes",
    "estimate",
    "lora_param_count",
    "module_shapes",
    "steps_for_epochs",
]

PLAN_BASE_PARAMS: Final = 12.2e9  # the plan's rounding of Mistral-Nemo (§7)
# fp32 LoRA weight + fp32 gradient + two fp32 Adam moments: peft keeps LoRA weights in fp32 on a
# k-bit model (the plan's formula counts a 2-byte weight; D-114)
LORA_BYTES_PER_PARAM: Final = 4 + 4 + 8
# nf4 with double quantization: 4 bits + absmax (8 bits per 64) + second level (32 bits per
# 64 × 256), in bytes per parameter
NF4_DOUBLE_QUANT_BYTES: Final = 0.5 + 1 / 64 + 4 / (64 * 256)
FP32: Final = 4
LOGIT_COPIES: Final = 3  # fp32 logits, their log-softmax and the gradient (rough)


def steps_for_epochs(n_train: int, micro: int, accum: int, epochs: int) -> int:
    """Optimizer steps of `epochs` full passes. HF Trainer's own count is
    ceil(ceil(n / micro) / accum), which equals ceil(n / (micro × accum))."""
    if min(n_train, micro, accum, epochs) <= 0:
        raise ValueError("n_train, micro, accum and epochs must be positive")
    return math.ceil(n_train / (micro * accum)) * epochs


@dataclass(frozen=True)
class MistralDims:
    """The shapes that matter for LoRA and memory, from a Mistral `config.json`."""

    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MistralDims | None:
        """None if the config is not a Mistral-like causal LM config (e.g. a test stub)."""
        try:
            heads = int(config["num_attention_heads"])
            hidden = int(config["hidden_size"])
            return cls(hidden=hidden, intermediate=int(config["intermediate_size"]),
                       layers=int(config["num_hidden_layers"]), heads=heads,
                       kv_heads=int(config.get("num_key_value_heads", heads)),
                       head_dim=int(config.get("head_dim") or hidden // heads),
                       vocab=int(config["vocab_size"]),
                       tied_embeddings=bool(config.get("tie_word_embeddings", False)))  # fmt: skip
        except (KeyError, TypeError, ValueError):
            return None


def module_shapes(dims: MistralDims) -> dict[str, tuple[int, int]]:
    """(in_features, out_features) of the linear layers of one decoder layer."""
    q_out, kv_out = dims.heads * dims.head_dim, dims.kv_heads * dims.head_dim
    return {
        "q_proj": (dims.hidden, q_out),
        "k_proj": (dims.hidden, kv_out),
        "v_proj": (dims.hidden, kv_out),
        "o_proj": (q_out, dims.hidden),
        "gate_proj": (dims.hidden, dims.intermediate),
        "up_proj": (dims.hidden, dims.intermediate),
        "down_proj": (dims.intermediate, dims.hidden),
    }


def lora_param_count(dims: MistralDims, target_modules: Sequence[str], rank: int) -> int:
    """layers × r × Σ(in + out) over the target modules (A is r × in, B is out × r)."""
    shapes = module_shapes(dims)
    unknown = sorted(set(target_modules) - set(shapes))
    if unknown:
        raise ValueError(f"unknown target modules {unknown}; known: {sorted(shapes)}")
    return dims.layers * rank * sum(sum(shapes[name]) for name in target_modules)


def _layer_linear_params(dims: MistralDims) -> int:
    return dims.layers * sum(i * o for i, o in module_shapes(dims).values())


def base_weight_bytes(dims: MistralDims) -> int:
    """nf4 linear layers + embeddings, lm_head and norms in fp32: bitsandbytes leaves them
    unquantized and `prepare_model_for_kbit_training` casts them to fp32."""
    embeddings = dims.vocab * dims.hidden * (1 if dims.tied_embeddings else 2)
    norms = (2 * dims.layers + 1) * dims.hidden
    return round(_layer_linear_params(dims) * NF4_DOUBLE_QUANT_BYTES + (embeddings + norms) * FP32)


def estimate(cfg: TrainConfig, stats: TokenStats, tokens_per_s: float | None,
             dims: MistralDims | None) -> Estimate:  # fmt: skip
    """The estimate of one training run of `cfg` over the records of `stats` (plan step 12)."""
    t, budget = cfg.training, cfg.budget
    per_epoch = steps_for_epochs(stats.records, t.micro_batch_size,
                                 t.gradient_accumulation_steps, 1)  # fmt: skip
    full = per_epoch * t.epochs
    steps = min(full, t.max_steps)
    processed = round(stats.tokens_total * steps / per_epoch)
    trainable = round(stats.tokens_trainable * steps / per_epoch)
    gpu_hours = processed / tokens_per_s / 3600 if tokens_per_s else None
    rate = budget.gpu_hourly_rate
    cost = gpu_hours * rate * budget.overhead_factor if gpu_hours is not None and rate else None
    wall_hours = t.max_wall_time_minutes / 60
    notes = []
    if tokens_per_s is None:
        notes.append("tokens/s not measured yet: GPU hours and cost come after the smoke run "
                     "on the GPU (plan step 13)")  # fmt: skip
    if steps < full:
        notes.append(f"max_steps {t.max_steps} < {full} steps of {t.epochs} epoch(s): a partial "
                     "pass over the data")  # fmt: skip
    lora = lora_param_count(dims, cfg.lora.target_modules, cfg.lora.rank) if dims else None
    memory = MemoryEstimate(
        base_weights_plan_bytes=round(PLAN_BASE_PARAMS * 0.5 * 1.1),
        base_weights_bytes=base_weight_bytes(dims) if dims else None,
        lora_bytes=lora * LORA_BYTES_PER_PARAM if lora is not None else None,
        logits_bytes=stats.tokens_max * dims.vocab * FP32 * LOGIT_COPIES if dims else None,
    )
    return Estimate(
        records=stats.records, micro_batch_size=t.micro_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps, epochs=t.epochs,
        steps_per_epoch=per_epoch, steps_for_epochs=full, max_steps=t.max_steps,
        optimizer_steps=steps, processed_tokens=processed, trainable_tokens=trainable,
        tokens_per_s=tokens_per_s, gpu_hours=gpu_hours, overhead_factor=budget.overhead_factor,
        hourly_rate=rate, currency=budget.currency, cost=cost,
        wall_time_limit_hours=wall_hours, cost_ceiling=wall_hours * rate if rate else None,
        lora_trainable_params=lora, memory=memory, notes=notes,
    )  # fmt: skip
