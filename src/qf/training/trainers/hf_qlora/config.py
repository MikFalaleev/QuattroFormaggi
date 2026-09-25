"""Parameters of the `hf_trainer_qlora` adapter (light: no heavy imports, plan C.5)."""

from __future__ import annotations

from typing import Literal

from qf.common import StrictConfig

__all__ = ["HFQLoRATrainerConfig"]


class HFQLoRATrainerConfig(StrictConfig):
    # cuda_4bit: the pinned base in nf4 on one NVIDIA GPU (QLoRA); tiny_random_cpu: a randomly
    # initialised 2-layer Mistral on CPU for tests and rehearsals (never a real model)
    loader: Literal["cuda_4bit", "tiny_random_cpu"] = "cuda_4bit"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
