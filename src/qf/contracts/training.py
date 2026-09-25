"""Training contracts (plan step 12): the QLoRA config, token statistics and the estimate.

`TrainConfig` lives here, not in `qf.training`, because the `AdapterTrainer` port takes it
(every type of a port signature is declared in `qf.contracts`, plan C.4; D-114). The training
stage loads it from YAML and runs the preflight checks (`qf.training.config`).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qf.common import ComponentConfig, StrictConfig

__all__ = [
    "PLACEHOLDER_REVISION",
    "Preflight",
    "DataHashes",
    "Estimate",
    "ExperimentSpec",
    "MemoryEstimate",
    "TokenStats",
    "TrainBudget",
    "TrainConfig",
    "TrainData",
    "TrainLogging",
    "TrainLora",
    "TrainModel",
    "TrainOutput",
    "TrainQuantization",
    "TrainSettings",
]

PLACEHOLDER_REVISION: Final = "REQUIRED_PINNED_COMMIT"
_SHA256 = r"^[0-9a-f]{64}$"


class TrainModel(StrictConfig):
    id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str
    # the pinned tokenizer and weights files (`qf tokens fetch`, `qf train fetch-base`)
    base_model_config: Path = Path("configs/train/base_model.yaml")
    preserve_tokenizer: Literal[True] = True  # the vocabulary and template are never changed
    preserve_chat_template: Literal[True] = True

    @field_validator("revision")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if value == PLACEHOLDER_REVISION:
            raise ValueError(f"{PLACEHOLDER_REVISION} is a placeholder: pin a full commit sha")
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("must be a full 40-hex commit sha, not a branch or tag")
        return value


class TrainQuantization(StrictConfig):
    load_in_4bit: bool
    type: Literal["nf4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bf16", "fp16"]

    @field_validator("compute_dtype", mode="before")
    @classmethod
    def _explicit_dtype(cls, value: Any) -> Any:
        if value == "bfloat16_if_supported":
            raise ValueError("decide explicitly: bf16 or fp16 by torch.cuda.is_bf16_supported() "
                             "on the GPU machine (plan step 13), not at run time")  # fmt: skip
        return value


class TrainLora(StrictConfig):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)


class DataHashes(StrictConfig):
    train: str = Field(pattern=_SHA256)
    val: str = Field(pattern=_SHA256)


class TrainData(StrictConfig):
    train_path: Path  # sft_dataset artifacts, relative to the project root
    val_path: Path
    expected_sha256: DataHashes
    max_skipped_share: float = Field(default=0.01, ge=0.0, le=1.0)  # records too long to fit


class TrainSettings(StrictConfig):
    max_sequence_length: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    epochs: int = Field(gt=0)
    # required: a positive max_steps overrides epochs in HF Trainer (plan step 12)
    max_steps: int = Field(gt=0)
    max_wall_time_minutes: int = Field(gt=0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    lr_scheduler: Literal["cosine"] = "cosine"
    max_grad_norm: float = Field(gt=0)
    gradient_checkpointing: bool = True
    use_cache: Literal[False] = False
    packing: Literal[False] = False
    loss_scope: Literal["assistant_tokens_only"] = "assistant_tokens_only"
    checkpoint_every_optimizer_steps: int = Field(gt=0)
    evaluate_every_optimizer_steps: int = Field(gt=0)
    keep_checkpoints: int = Field(gt=0)
    optimizer: Literal["adamw_torch"] = "adamw_torch"


class TrainLogging(StrictConfig):
    external_reporting: Literal[False] = False
    record_peak_memory: bool = True


class TrainBudget(StrictConfig):
    gpu_hourly_rate: float | None = Field(default=None, gt=0)
    # measured on the GPU machine (plan step 13); None until then
    measured_tokens_per_s: float | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, gt=0)
    currency: str = "RUB"
    # evaluation, model loading and checkpoints on top of the training loop (plan step 12)
    overhead_factor: float = Field(default=1.3, ge=1.0)


class TrainOutput(StrictConfig):
    run_root: Path = Path("runs")


class TrainConfig(StrictConfig):
    """`configs/train/*.yaml`: the schema of the development plan §10, pinned (plan step 12)."""

    run_name: str = Field(min_length=1)
    seed: int
    model: TrainModel
    quantization: TrainQuantization
    lora: TrainLora
    data: TrainData
    training: TrainSettings
    logging: TrainLogging = TrainLogging()
    budget: TrainBudget = TrainBudget()
    output: TrainOutput = TrainOutput()
    trainer: ComponentConfig  # an open {name, **params} selector (plan C.5)


class _Computed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenStats(_Computed):
    """Tokens of the training records as the trainer sees them (variant-B features)."""

    records: int  # records that fit into max_sequence_length
    skipped_too_long: int
    tokens_total: int  # every token processed, prompt included
    tokens_trainable: int  # answer tokens + EOS (the loss mask)
    tokens_max: int
    tokens_mean: float


class MemoryEstimate(_Computed):
    """Rough GPU memory, bytes; to be checked by measurement (plan step 13)."""

    base_weights_plan_bytes: int  # the plan's formula: 12.2e9 × 0.5 B × 1.1
    base_weights_bytes: int | None  # 4-bit layers + fp32 embeddings and lm_head (by config.json)
    lora_bytes: int | None  # LoRA weights, gradients and Adam states, fp32
    logits_bytes: int | None  # fp32 logits of the longest record, forward + backward
    activations_bytes: None = None  # unknown until measured
    note: str = "rough; check by measurement (plan step 13)"


class Estimate(_Computed):
    records: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    steps_per_epoch: int
    steps_for_epochs: int  # ceil(records / (micro × accum)) × epochs
    max_steps: int
    optimizer_steps: int  # min(steps_for_epochs, max_steps)
    processed_tokens: int  # all tokens of the optimizer steps, not only the trainable ones
    trainable_tokens: int
    tokens_per_s: float | None
    gpu_hours: float | None  # None until tokens/s is measured (plan step 13)
    overhead_factor: float
    hourly_rate: float | None
    currency: str
    cost: float | None  # gpu_hours × rate × overhead
    wall_time_limit_hours: float
    cost_ceiling: float | None  # max_wall_time_minutes × rate: the most one run can cost
    lora_trainable_params: int | None
    memory: MemoryEstimate
    notes: list[str] = Field(default_factory=list)


class ExperimentSpec(StrictConfig):
    """`configs/experiments/EXXX.yaml`: an experiment registered in Git before it runs (D-118).

    The training config is fixed by its hash (without the seed): a run under this ID with other
    parameters is refused. `seeds` lists the allowed replicates of the same config."""

    id: str = Field(pattern=r"^E\d{3}$")
    title: str = Field(min_length=1)
    registered: date
    plan_step: str = Field(min_length=1)
    question: str = Field(min_length=1)
    hypotheses: list[str] = Field(min_length=1)
    success: str = Field(min_length=1)
    config: Path
    config_sha256: str = Field(pattern=_SHA256)
    seeds: list[int] = Field(min_length=1)
    requires_approval: bool = True
    notes: str = ""

    @field_validator("seeds")
    @classmethod
    def _distinct(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("seeds must be distinct")
        return value


class Preflight(_Computed):
    """One line of `preflight.jsonl` in a run directory, written by `qf train run` before any
    model is loaded (one line per session: the start and every resume)."""

    session: int = Field(ge=1)
    created_at: datetime
    run_id: str
    experiment_id: str | None
    seed: int
    config_sha256: str  # the training config without its seed (see ExperimentSpec)
    approval: str | None  # the human decision that allowed this session, quoted
    allow_code_change: str | None = None  # resume only: why a different commit is accepted
    resume_from: str | None = None
    git_commit: str | None
    git_dirty: bool
    host: str  # a non-identifying machine label (qf.common.machine_label)
    checks: list[str]  # the preflight checks that passed
    token_stats: TokenStats
    estimate: Estimate
