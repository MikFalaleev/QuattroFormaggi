"""The training config and the checks before a run (plan step 12), independent of the trainer.

The schema itself is `qf.contracts.TrainConfig` (the `AdapterTrainer` port takes it, D-114).
Here: loading, command-line overrides, and the preflight refusals of `qf train run` — data
hashes, the pinned base model, `max_steps` above the planned epochs, and the budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from qf.common import (
    ArtifactRef,
    QFError,
    format_validation_error,
    load_yaml_config,
    read_artifact,
    sha256_file,
    sha256_text,
)
from qf.contracts import Estimate, SFTRecord, TrainConfig, supported_versions
from qf.domain import parse_sft_jsonl
from qf.training.estimate import steps_for_epochs
from qf.training.tokenizer_io import BaseModelConfig

__all__ = [
    "TRAIN_CONFIG_COPY",
    "base_model_for",
    "check_budget",
    "check_max_steps",
    "data_refs",
    "load_records",
    "load_train_config",
    "system_prompt_hash",
    "with_overrides",
]

# the config a run was started with, in its run directory (resume continues with exactly it)
TRAIN_CONFIG_COPY: Final = "train_config.json"


def load_train_config(path: Path) -> TrainConfig:
    return load_yaml_config(path, TrainConfig)


def with_overrides(cfg: TrainConfig, **training: Any) -> TrainConfig:
    """`cfg` with `training.<key>` replaced for the keys whose value is not None, validated."""
    changes = {key: value for key, value in training.items() if value is not None}
    if not changes:
        return cfg
    data = cfg.model_dump(mode="python")
    data["training"].update(changes)
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        raise QFError(f"invalid override: {format_validation_error(exc)}") from exc


def data_refs(cfg: TrainConfig, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    """The verified train and val `sft_dataset` artifacts; their sha256 must equal the config."""
    refs = []
    hashes = cfg.data.expected_sha256
    for split, path, expected in (("train", cfg.data.train_path, hashes.train),
                                  ("val", cfg.data.val_path, hashes.val)):  # fmt: skip
        ref = read_artifact(path, "sft_dataset", supported_versions("sft_dataset"), root=root)
        if ref.sha256 != expected:
            raise QFError(f"data hash mismatch for {split} ({path}): config expects "
                          f"{expected[:12]}…, the file is {ref.sha256[:12]}…")  # fmt: skip
        refs.append(ref)
    return refs[0], refs[1]


def load_records(path: Path, expected_sha256: str) -> list[SFTRecord]:
    """Records of a JSONL file whose sha256 is checked before it is read."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise QFError(f"{path}: sha256 {actual[:12]}… differs from the expected "
                      f"{expected_sha256[:12]}…")  # fmt: skip
    records, issues = parse_sft_jsonl(path.read_text(encoding="utf-8"), source=path.name)
    if issues:
        raise QFError(f"{path}: {len(issues)} invalid record(s), e.g. {issues[0]}")
    return records


def base_model_for(cfg: TrainConfig, root: Path) -> BaseModelConfig:
    """The pinned base-model files config; it must name the same model and revision."""
    base = load_yaml_config(root / cfg.model.base_model_config, BaseModelConfig)
    if (base.id, base.revision) != (cfg.model.id, cfg.model.revision):
        raise QFError(f"{cfg.model.base_model_config} pins {base.id}@{base.revision[:12]}, the "
                      f"training config {cfg.model.id}@{cfg.model.revision[:12]}")  # fmt: skip
    return base


def system_prompt_hash(records: Sequence[SFTRecord]) -> str:
    """sha256 of the one system prompt of the records (a mix of prompts is refused)."""
    prompts = {r.messages[0].content for r in records}
    if len(prompts) != 1:
        raise QFError(f"expected one system prompt in the training data, found {len(prompts)}")
    return sha256_text(prompts.pop())


def check_max_steps(cfg: TrainConfig, n_train: int) -> None:
    """A positive `max_steps` overrides epochs in HF Trainer, which then repeats the data until
    it reaches them: more than the planned epochs is refused (plan step 12)."""
    t = cfg.training
    planned = steps_for_epochs(n_train, t.micro_batch_size, t.gradient_accumulation_steps,
                               t.epochs)  # fmt: skip
    if t.max_steps > planned:
        raise QFError(f"training.max_steps {t.max_steps} exceeds {planned} = ceil({n_train} / "
                      f"({t.micro_batch_size} × {t.gradient_accumulation_steps})) × {t.epochs} "
                      f"epoch(s); set max_steps to {planned} for exactly {t.epochs} epoch(s) and "
                      "keep any reserve in max_wall_time_minutes")  # fmt: skip


def check_budget(estimate: Estimate, cfg: TrainConfig) -> None:
    """Refuse a run whose cost may exceed `budget.max_cost`: the wall-time ceiling always, the
    throughput-based cost when tokens/s is known."""
    limit = cfg.budget.max_cost
    if limit is None:
        return
    currency = cfg.budget.currency
    if estimate.cost_ceiling is None:
        raise QFError("budget.max_cost is set but budget.gpu_hourly_rate is not")
    if estimate.cost_ceiling > limit:
        raise QFError(f"over budget: max_wall_time_minutes {cfg.training.max_wall_time_minutes} × "
                      f"{estimate.hourly_rate} {currency}/h = {estimate.cost_ceiling:.2f} "
                      f"{currency} > budget.max_cost {limit:.2f} {currency}")  # fmt: skip
    if estimate.cost is not None and estimate.cost > limit:
        raise QFError(f"over budget: estimated cost {estimate.cost:.2f} {currency} > "
                      f"budget.max_cost {limit:.2f} {currency}")  # fmt: skip
