"""Parameters of the `hf_trainer_qlora` adapter (light: no heavy imports, plan C.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qf.common import StrictConfig

__all__ = ["HFQLoRATrainerConfig"]


class HFQLoRATrainerConfig(StrictConfig):
    # cuda_4bit: the pinned base in nf4 on one NVIDIA GPU (QLoRA); tiny_random_cpu: a randomly
    # initialised 2-layer Mistral on CPU for tests and rehearsals (never a real model)
    loader: Literal["cuda_4bit", "tiny_random_cpu"] = "cuda_4bit"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    monitor_interval_s: float = Field(default=15.0, gt=0)  # resources.jsonl (D-118)
    # sample answers after training, with and without the adapter — not an evaluation (D-118)
    sample_records: int = Field(default=5, ge=0)
    sample_max_new_tokens: int = Field(default=512, gt=0)
