"""`hf_trainer_qlora`: LoRA adapter training with `transformers.Trainer` + peft (plan step 12).

One code path for the real QLoRA run and the tiny CPU tests: the loader (`model_loading`)
decides device and precision. The Trainer restores optimizer, scheduler, RNG and data position
on resume. Every run keeps `run_manifest.json` (updated on each log line), `train_log.jsonl`
(train and eval lines; a train line lists the records of its optimizer step) and checkpoints;
a completed run also writes `adapter/` as a `lora_adapter@peft_lora_v1` artifact.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from qf.common import (
    ArtifactRef,
    QFError,
    RunManifest,
    atomic_write_text,
    collect_git_info,
    collect_hardware,
    collect_package_versions,
    project_root,
    read_manifest,
    set_global_seed,
    update_manifest,
    write_artifact,
    write_manifest,
)
from qf.contracts import Estimate, TokenStats, TrainConfig
from qf.training import (
    TRAIN_CONFIG_COPY,
    BaseModelConfig,
    MistralDims,
    PadCollator,
    base_model_for,
    chat_template_hash,
    choose_pad_token,
    estimate,
    load_records,
    system_prompt_hash,
    tokenizer_hash,
    verify_tokenizer_files,
)
from qf.training.trainers.hf_qlora.callbacks import (
    BatchLog,
    JsonlLoggerCallback,
    ManifestCallback,
    NanGuardCallback,
    RunState,
    WallTimeLimitCallback,
)
from qf.training.trainers.hf_qlora.config import HFQLoRATrainerConfig
from qf.training.trainers.hf_qlora.dataset import RECORD_INDEX, IndexedCollator, load_features
from qf.training.trainers.hf_qlora.model_loading import LOADERS, apply_lora, count_trainable

__all__ = [
    "ADAPTER_DIR",
    "ADAPTER_MANIFEST",
    "ADAPTER_SCHEMA",
    "TEST_MODEL_BASE",
    "TRAIN_LOG",
    "HFQLoRATrainer",
    "QFTrainer",
    "Throughput",
    "TrainingStopped",
    "interrupt_on_termination",
    "name_base_model",
    "run_training",
]

ADAPTER_DIR: Final = "adapter"
ADAPTER_SCHEMA: Final = "peft_lora_v1"
ADAPTER_MANIFEST: Final = "qf_adapter_manifest.json"
TRAIN_LOG: Final = "train_log.jsonl"
PACKAGES: Final = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers",
                   "safetensors")  # fmt: skip


class TrainingStopped(QFError):
    """The run stopped before max_steps (wall time, NaN, interruption); a checkpoint is kept."""


@dataclass
class Throughput:
    """Tokens and seconds of the optimizer steps only (no evaluation, no checkpoint saving)."""

    tokens: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_s(self) -> float | None:
        return self.tokens / self.seconds if self.seconds > 0 else None


class QFTrainer(Trainer):
    """Removes `record_index` before the forward pass and records what training saw."""

    def __init__(self, *args: Any, batches: BatchLog, throughput: Throughput,
                 **kwargs: Any) -> None:  # fmt: skip
        super().__init__(*args, **kwargs)
        self.batches = batches
        self.throughput = throughput

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: Any = None) -> Any:  # fmt: skip
        indexes = inputs.pop(RECORD_INDEX, None)
        if model.training and indexes is not None:
            self.batches.add([int(i) for i in indexes.tolist()])
            self.throughput.tokens += int(inputs["attention_mask"].sum())
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)  # fmt: skip


class _StepTimer(TrainerCallback):
    def __init__(self, throughput: Throughput) -> None:
        self.throughput = throughput
        self.started = 0.0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.throughput.seconds += time.perf_counter() - self.started


@contextmanager
def interrupt_on_termination() -> Iterator[None]:
    """SIGTERM and SIGHUP (a `kill`, a dropped SSH session) end training like Ctrl-C does, so
    the run is recorded as aborted with its last checkpoint instead of dying silently."""

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _stored_config(run_dir: Path) -> dict[str, Any]:
    try:
        stored: dict[str, Any] = json.loads((run_dir / TRAIN_CONFIG_COPY).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise QFError(f"{run_dir} has no readable {TRAIN_CONFIG_COPY}: not a training run") from exc
    return stored


def _config_copy(cfg: TrainConfig, params: HFQLoRATrainerConfig) -> dict[str, Any]:
    return {"train": cfg.model_dump(mode="json"), "trainer": params.model_dump(mode="json")}


def _check_resume(cfg: TrainConfig, params: HFQLoRATrainerConfig, run_dir: Path,
                  checkpoint: Path) -> None:  # fmt: skip
    if checkpoint.parent.resolve() != run_dir.resolve() or not checkpoint.name.startswith(
        "checkpoint-"
    ):
        raise QFError(f"{checkpoint} is not a checkpoint-<step> directory of {run_dir}")
    if not checkpoint.is_dir():
        raise QFError(f"checkpoint {checkpoint} does not exist")
    if _stored_config(run_dir) != _config_copy(cfg, params):
        raise QFError(f"the config differs from the one {run_dir.name} was started with: resume "
                      "continues the same run with the same settings")  # fmt: skip
    if read_manifest(run_dir).status == "completed":
        raise QFError(f"{run_dir.name} is already completed")


def _last_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return checkpoints[-1] if checkpoints else None


def _training_arguments(cfg: TrainConfig, run_dir: Path, device: str,
                        precision: str) -> TrainingArguments:  # fmt: skip
    t = cfg.training
    return TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=t.micro_batch_size,
        per_device_eval_batch_size=t.micro_batch_size,  # logits over 131k tokens are large
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_steps=t.warmup_ratio,  # a float below 1 is a share of the steps (transformers 5)
        max_grad_norm=t.max_grad_norm,
        max_steps=t.max_steps,
        num_train_epochs=t.epochs,  # overridden by max_steps; checked by `check_max_steps`
        optim=t.optimizer,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        use_cpu=device == "cpu",  # never MPS: the CPU path is for tests and rehearsals
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_nan_inf_filter=False,  # a NaN loss must reach the NaN guard, not be averaged
        eval_strategy="steps",
        eval_steps=t.evaluate_every_optimizer_steps,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=t.checkpoint_every_optimizer_steps,
        save_total_limit=t.keep_checkpoints,
        save_only_model=False,  # optimizer, scheduler and RNG states for resume
        ignore_data_skip=False,  # resume skips the batches already trained on
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )  # fmt: skip


def _base_dims(base: BaseModelConfig, root: Path) -> MistralDims | None:
    path = base.directory(root) / "config.json"
    try:
        return MistralDims.from_config(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def run_training(cfg: TrainConfig, params: HFQLoRATrainerConfig, train_ref: ArtifactRef,
                 val_ref: ArtifactRef, run_dir: Path, resume_from: Path | None, *,
                 root: Path | None = None) -> ArtifactRef:  # fmt: skip
    root = root if root is not None else project_root()
    started = time.monotonic()
    if resume_from is not None:
        _check_resume(cfg, params, run_dir, resume_from)
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise QFError(f"{run_dir} is not empty: a new run needs a new directory")
    set_global_seed(cfg.seed)
    base = base_model_for(cfg, root)
    verify_tokenizer_files(base, root)
    loaded = LOADERS[params.loader](cfg, base, root, params)
    tok = loaded.tokenizer
    vocab_size = len(tok)
    t = cfg.training
    share = cfg.data.max_skipped_share
    train_ds, train_skipped = load_features(root / train_ref.path, tok, t.max_sequence_length,
                                            train_ref.sha256, max_skipped_share=share)  # fmt: skip
    val_ds, val_skipped = load_features(root / val_ref.path, tok, t.max_sequence_length,
                                        val_ref.sha256, max_skipped_share=share)  # fmt: skip
    prompt_sha = system_prompt_hash(
        load_records(root / train_ref.path, train_ref.sha256)
        + load_records(root / val_ref.path, val_ref.sha256)
    )
    model = apply_lora(loaded.model, cfg.lora)
    trainable, total = count_trainable(model)
    if len(tok) != vocab_size:
        raise QFError("the vocabulary changed while preparing training")
    pad_id, pad_reason = choose_pad_token(tok)
    setup = {"trainable_params": trainable, "total_params": total,
             "train_records": len(train_ds), "val_records": len(val_ds),
             "train_skipped_too_long": len(train_skipped),
             "val_skipped_too_long": len(val_skipped), "train_tokens": train_ds.tokens_total,
             "vocab_size": vocab_size, "pad_token_id": pad_id, "pad_token_reason": pad_reason,
             "device": loaded.device, "precision": loaded.precision,
             "quantization": loaded.quantization}  # fmt: skip
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.3%})")
    if resume_from is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(run_dir / TRAIN_CONFIG_COPY,
                          json.dumps(_config_copy(cfg, params), indent=2) + "\n")  # fmt: skip
        git_commit, git_dirty = collect_git_info(root)
        write_manifest(RunManifest(
            run_id=run_dir.name, kind="train", created_at=datetime.now(UTC), status="running",
            git_commit=git_commit, git_dirty=git_dirty,
            hardware={**collect_hardware(), **loaded.hardware},
            package_versions=collect_package_versions(("quattro-formaggi", *PACKAGES)),
            base_revision=cfg.model.revision,
            data_hashes={train_ref.path.as_posix(): train_ref.sha256,
                         val_ref.path.as_posix(): val_ref.sha256},
            tokenizer_hash=tokenizer_hash(base.directory(root)),
            chat_template_hash=chat_template_hash(tok), prompt_hashes={"system": prompt_sha},
            seed=cfg.seed, config=_config_copy(cfg, params), metrics=setup,
        ), run_dir)  # fmt: skip
    else:
        previous = read_manifest(run_dir)
        sessions = [*previous.metrics.get("resumed_from", []), _relative(resume_from, root)]
        update_manifest(run_dir, status="running",
                        parent_checkpoint=_relative(resume_from, root),
                        metrics={**previous.metrics, **setup,
                                 "resumed_from": sessions})  # fmt: skip
    state, throughput = RunState(), Throughput()
    batches = BatchLog(train_ds.record_ids)
    trainer = QFTrainer(
        model=model, args=_training_arguments(cfg, run_dir, loaded.device, loaded.precision),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=IndexedCollator(PadCollator(pad_token_id=pad_id)),
        callbacks=[NanGuardCallback(state), WallTimeLimitCallback(t.max_wall_time_minutes, state),
                   _StepTimer(throughput), JsonlLoggerCallback(run_dir / TRAIN_LOG, batches),
                   ManifestCallback(run_dir)],
        batches=batches, throughput=throughput,
    )  # fmt: skip
    try:
        with interrupt_on_termination():
            output = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except KeyboardInterrupt as exc:
        state.stop("aborted", f"interrupted ({exc or 'SIGINT'})")
        output = None
    except Exception as exc:
        _finish(run_dir, root, started, loaded.device, throughput, "failed",
                {"stop_reason": f"{type(exc).__name__}: {exc}"})  # fmt: skip
        raise
    session = {"session_optimizer_steps_seconds": round(throughput.seconds, 3),
               "session_tokens": throughput.tokens,
               "session_tokens_per_s": throughput.tokens_per_s,
               "global_step": trainer.state.global_step,
               **(output.metrics if output is not None else {})}  # fmt: skip
    if state.status != "running":
        last = _last_checkpoint(run_dir)
        where = _relative(last, root) if last else None
        _finish(run_dir, root, started, loaded.device, throughput, state.status,
                {**session, "stop_reason": state.reason, "last_checkpoint": where})  # fmt: skip
        raise TrainingStopped(f"training {state.status} at step {trainer.state.global_step}: "
                              f"{state.reason}; last checkpoint: {where} (continue with "
                              f"`qf train run --resume {where}`)")  # fmt: skip
    step = trainer.state.global_step
    evals = [e for e in trainer.state.log_history if "eval_loss" in e and e.get("step") == step]
    final_eval = evals[-1] if evals else trainer.evaluate()
    adapter_dir = run_dir / ADAPTER_DIR
    trainer.save_model(str(adapter_dir))
    trainer.save_state()
    name_base_model(adapter_dir, loaded.base_model, cfg.model.revision)
    adapter_manifest = {
        "base_model": cfg.model.id, "revision": cfg.model.revision,
        "tokenizer_hash": tokenizer_hash(base.directory(root)),
        "chat_template_hash": chat_template_hash(tok), "system_prompt_sha256": prompt_sha,
        "data": {"train": train_ref.sha256, "val": val_ref.sha256}, "run_id": run_dir.name,
        "global_step": step, "lora": cfg.lora.model_dump(mode="json"),
        "trainable_params": trainable, "loader": params.loader,
        "quantization": loaded.quantization, "trained_on": loaded.base_model or TEST_MODEL_BASE,
    }  # fmt: skip
    atomic_write_text(adapter_dir / ADAPTER_MANIFEST, json.dumps(adapter_manifest, indent=2) + "\n")
    ref = write_artifact(adapter_dir, "lora_adapter", ADAPTER_SCHEMA, run_dir.name,
                         parents=(train_ref.sha256, val_ref.sha256),
                         extra={"base_model": loaded.base_model or TEST_MODEL_BASE,
                                "revision": cfg.model.revision, "global_step": step,
                                "loader": params.loader},
                         root=root)  # fmt: skip
    _finish(run_dir, root, started, loaded.device, throughput, "completed",
            {**session, "final_eval_loss": final_eval.get("eval_loss"),
             "adapter": ref.path.as_posix(), "adapter_sha256": ref.sha256})  # fmt: skip
    return ref


TEST_MODEL_BASE: Final = "tiny random Mistral (a test model, not a base model)"


def name_base_model(adapter_dir: Path, base_model: str | None, revision: str) -> None:
    """The adapter names the pinned Hub model it was trained on, not the local directory; an
    adapter of the random test model says so instead. peft's generated README (a model-card
    template with the local path) is dropped: the model card is written at release (step 19)."""
    path = adapter_dir / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(base_model_name_or_path=base_model or TEST_MODEL_BASE,
                  revision=revision if base_model else None)  # fmt: skip
    atomic_write_text(path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    (adapter_dir / "README.md").unlink(missing_ok=True)


def _finish(run_dir: Path, root: Path, started: float, device: str, throughput: Throughput,
            status: Any, metrics: dict[str, Any]) -> None:  # fmt: skip
    previous = read_manifest(run_dir)
    wall = (previous.wall_time_s or 0.0) + time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    update_manifest(run_dir, status=status, wall_time_s=round(wall, 3), peak_memory_bytes=peak,
                    metrics={**previous.metrics, **metrics})  # fmt: skip


class HFQLoRATrainer:
    """The `AdapterTrainer` of HF Trainer + peft (+ bitsandbytes for the 4-bit base)."""

    Config = HFQLoRATrainerConfig

    def __init__(self, config: HFQLoRATrainerConfig) -> None:
        self.config = config

    def estimate(self, cfg: TrainConfig, token_stats: TokenStats,
                 measured_tokens_per_s: float | None) -> Estimate:  # fmt: skip
        root = project_root()
        return estimate(cfg, token_stats, measured_tokens_per_s,
                        _base_dims(base_model_for(cfg, root), root))  # fmt: skip

    def train(self, cfg: TrainConfig, train: ArtifactRef, val: ArtifactRef, run_dir: Path,
              resume: Path | None) -> ArtifactRef:  # fmt: skip
        return run_training(cfg, self.config, train, val, run_dir, resume)
