"""`qf train …` (plan step 12): estimate, training run, base weights, run verification.

`train run` refuses before loading any model when the data hashes differ, `max_steps` is above
the planned epochs or the cost may exceed `budget.max_cost`. The experiment protocol (D-118):
a paid run (a config with `budget.gpu_hourly_rate`) must be a registered experiment
(`--experiment`, exactly its config, a registered seed, a clean git tree) and carry the human
approval (`--approval`); `preflight.jsonl` is written before any model is loaded and the console
is copied to `console.log`. `train fetch-base` downloads ~24.5 GB and needs `--approval` too.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TextIO

from qf.cli.wiring import build
from qf.common import (
    RUNS,
    QFError,
    atomic_write_text,
    collect_git_info,
    load_yaml_config,
    machine_label,
    new_run_id,
    project_root,
    start_run,
)
from qf.contracts import Estimate, Preflight, TokenStats, TrainConfig
from qf.training import (
    CONSOLE_FILE,
    TRAIN_CONFIG_COPY,
    TRAINERS,
    BaseModelConfig,
    append_preflight,
    base_model_for,
    check_budget,
    check_max_steps,
    data_refs,
    download_base_weights,
    experiment_config_hash,
    load_experiment,
    load_records,
    load_tokenizer,
    load_train_config,
    read_preflights,
    registered_config,
    replicates_done,
    token_stats,
    verify_base_weights,
    verify_run,
    verify_tokenizer_files,
    with_overrides,
)

__all__ = [
    "configure_estimate",
    "configure_fetch_base",
    "configure_run",
    "configure_verify",
    "run_estimate",
    "run_fetch_base",
    "run_run",
    "run_verify",
]

DEFAULT_CONFIG: Final = Path("configs/train/qlora_nemo_v0.1.yaml")
DEFAULT_BASE_CONFIG: Final = Path("configs/train/base_model.yaml")
# assumptions for the table shown before tokens/s is measured, never a measurement
ASSUMED_TOKENS_PER_S: Final = (500, 1000, 2000, 3000)
GIB: Final = 2**30
MIN_APPROVAL_CHARS: Final = 12  # who, when and the quoted decision, not just "ok"


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    parser.add_argument("--tps", type=float, default=None,
                        help="measured tokens/s, all tokens (default: "
                             "budget.measured_tokens_per_s)")  # fmt: skip


def _add_experiment(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", default=None, metavar="EXXX",
                        help="a registered experiment (configs/experiments/EXXX.yaml): its "
                             "config exactly, no overrides")  # fmt: skip
    parser.add_argument("--seed", type=int, default=None,
                        help="one of the experiment's registered seeds (replicates)")  # fmt: skip


def _add_approval(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval", default=None, metavar="TEXT",
                        help="the human decision allowing this paid step: who, when, quote "
                             "(recorded in the run)")  # fmt: skip


def configure_estimate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help=f"training config (default {DEFAULT_CONFIG})")  # fmt: skip
    _add_experiment(parser)
    _add_overrides(parser)
    parser.add_argument("--rate", type=float, default=None,
                        help="GPU price per hour; default budget.gpu_hourly_rate")  # fmt: skip


def configure_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help=f"training config (default {DEFAULT_CONFIG})")  # fmt: skip
    parser.add_argument("--resume", type=Path, default=None, metavar="CHECKPOINT",
                        help="continue a stopped run from runs/<id>/checkpoint-<step> with the "
                             "config it was started with")  # fmt: skip
    _add_overrides(parser)
    parser.add_argument("--save-every", type=int, default=None,
                        help="override training.checkpoint_every_optimizer_steps")  # fmt: skip
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override training.evaluate_every_optimizer_steps")  # fmt: skip
    _add_experiment(parser)
    _add_approval(parser)
    parser.add_argument("--allow-code-change", default=None, metavar="WHY",
                        help="--resume only: accept a different code commit, "
                             "with the reason")  # fmt: skip


def configure_fetch_base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE_CONFIG,
                        help="base model config")  # fmt: skip
    _add_approval(parser)


def configure_verify(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", type=Path, help="a completed training run: runs/<id>")


def _stats(cfg: TrainConfig, root: Path) -> TokenStats:
    """Tokens of the train split as the trainer builds them (pinned tokenizer)."""
    base = base_model_for(cfg, root)
    verify_tokenizer_files(base, root)
    tok = load_tokenizer(base.directory(root), fix_mistral_regex=base.fix_mistral_regex)
    records = load_records(root / cfg.data.train_path, cfg.data.expected_sha256.train)
    stats, _ = token_stats(records, tok, cfg.training.max_sequence_length)
    return stats


def _preflight(cfg: TrainConfig, root: Path,
               tps: float | None) -> tuple[Any, TokenStats, Estimate]:  # fmt: skip
    data_refs(cfg, root)
    stats = _stats(cfg, root)
    check_max_steps(cfg, stats.records)
    trainer = build(TRAINERS, cfg.trainer)
    estimate = trainer.estimate(cfg, stats, tps or cfg.budget.measured_tokens_per_s)
    check_budget(estimate, cfg)
    return trainer, stats, estimate


def _n(value: int | float) -> str:
    """A number with thin spaces between thousands: 1 914 735."""
    return f"{value:,.0f}".replace(",", "\u2009")


def _gib(value: int | None) -> str:
    return "—" if value is None else f"{value / GIB:.1f} ГиБ"


def _money(value: float | None, currency: str) -> str:
    return "—" if value is None else f"{_n(value)} {currency}"


def render_estimate(cfg: TrainConfig, est: Estimate, stats: TokenStats) -> str:
    cur, rate = est.currency, est.hourly_rate
    lines = [
        f"# Оценка обучения: {cfg.run_name}", "",
        f"- Модель: `{cfg.model.id}@{cfg.model.revision[:12]}`, LoRA r={cfg.lora.rank}, "
        f"alpha={cfg.lora.alpha}, модули: {', '.join(cfg.lora.target_modules)}.",
        f"- Данные: `{cfg.data.train_path.as_posix()}` — {stats.records} записей "
        f"(не влезло в {cfg.training.max_sequence_length} токенов: {stats.skipped_too_long}), "
        f"{_n(stats.tokens_total)} токенов, обучаемых {_n(stats.tokens_trainable)} "
        f"({stats.tokens_trainable / max(stats.tokens_total, 1):.1%}); самая длинная запись "
        f"{stats.tokens_max} токенов.",
        "", "## Шаги", "",
        f"- Батч: {est.micro_batch_size} × накопление {est.gradient_accumulation_steps} = "
        f"{est.micro_batch_size * est.gradient_accumulation_steps} записей на шаг оптимизатора.",
        f"- Шагов на эпоху: {est.steps_per_epoch}; эпох: {est.epochs}; max_steps: "
        f"{est.max_steps}; будет выполнено шагов: **{est.optimizer_steps}**.",
        f"- Токенов пройдёт через модель: {_n(est.processed_tokens)} (обучаемых "
        f"{_n(est.trainable_tokens)}).",
        "", "## Время и стоимость", "",
    ]  # fmt: skip
    if est.tokens_per_s is None:
        lines.append("- Скорость (токенов/с) **не измерена**: время и стоимость появятся после "
                     "пробного прогона на GPU (шаг 13). Ниже — только предположения.")  # fmt: skip
    else:
        lines.append(f"- Измеренная скорость: {_n(est.tokens_per_s)} токенов/с → "
                     f"{est.gpu_hours:.2f} ч цикла обучения; с запасом ×{est.overhead_factor} "
                     f"стоимость {_money(est.cost, cur)}.")  # fmt: skip
    lines.append(f"- Жёсткий потолок одного запуска: лимит {est.wall_time_limit_hours:.1f} ч × "
                 f"тариф = {_money(est.cost_ceiling, cur)}.")  # fmt: skip
    if rate:
        lines += ["", f"Сценарии (**предположения, не замер**), тариф {rate} {cur}/ч, запас "
                      f"×{est.overhead_factor}:", "",
                  "| Скорость, токенов/с | Цикл обучения, ч | Стоимость |",
                  "|---:|---:|---:|"]  # fmt: skip
        for tps in ASSUMED_TOKENS_PER_S:
            hours = est.processed_tokens / tps / 3600
            lines.append(f"| {tps} | {hours:.2f} | "
                         f"{_money(hours * rate * est.overhead_factor, cur)} |")  # fmt: skip
    memory = est.memory
    lines += [
        "", "## Память GPU (грубо, проверить замером на шаге 13)", "",
        f"- Веса базы по формуле плана (12,2 млрд × 0,5 Б × 1,1): "
        f"{_gib(memory.base_weights_plan_bytes)}.",
        f"- Веса базы по config.json (nf4 + эмбеддинги и lm_head в fp32 после подготовки к "
        f"k-bit обучению): {_gib(memory.base_weights_bytes)}.",
        f"- LoRA: {_n(est.lora_trainable_params) if est.lora_trainable_params else '—'} "
        f"параметров; веса, градиенты и Adam в fp32: {_gib(memory.lora_bytes)}.",
        f"- Логиты самой длинной записи (fp32, ~3 копии): {_gib(memory.logits_bytes)}.",
        "- Активации: неизвестно до замера (включён gradient checkpointing).",
    ]  # fmt: skip
    if est.optimizer_steps < est.steps_for_epochs:
        lines += ["", f"Внимание: max_steps {est.max_steps} меньше {est.steps_for_epochs} шагов "
                      f"{est.epochs} эпох(и) — пройдёт только часть данных."]  # fmt: skip
    return "\n".join(lines) + "\n"


def _approval(text: str | None, what: str) -> str:
    if text is None or len(text.strip()) < MIN_APPROVAL_CHARS:
        raise QFError(f"{what} needs the human decision: --approval \"who, when, quote\" "
                      f"(at least {MIN_APPROVAL_CHARS} characters; D-118)")  # fmt: skip
    return text.strip()


def _selected_config(args: argparse.Namespace, root: Path, overrides: dict[str, Any]
                     ) -> tuple[TrainConfig, str | None]:  # fmt: skip
    """The config of a registered experiment (no overrides allowed) or of `--config`."""
    if args.experiment is None:
        if args.seed is not None:
            raise QFError("--seed chooses a registered replicate: use it with --experiment")
        cfg = with_overrides(load_train_config(root / (args.config or DEFAULT_CONFIG)), **overrides)
        return cfg, None
    if args.config is not None or any(v is not None for v in overrides.values()):
        raise QFError(f"{args.experiment} runs exactly its registered config: no --config or "
                      "overrides (new parameters need a new experiment)")  # fmt: skip
    spec = load_experiment(root, args.experiment)
    return registered_config(root, spec, args.seed), spec.id


def run_estimate(args: argparse.Namespace) -> int:
    root = project_root()
    cfg, _ = _selected_config(args, root, {"max_steps": args.max_steps})
    if args.rate is not None:
        cfg = cfg.model_copy(update={"budget": cfg.budget.model_copy(
            update={"gpu_hourly_rate": args.rate})})  # fmt: skip
    data_refs(cfg, root)
    stats = _stats(cfg, root)
    check_max_steps(cfg, stats.records)
    trainer = build(TRAINERS, cfg.trainer)
    est = trainer.estimate(cfg, stats, args.tps or cfg.budget.measured_tokens_per_s)
    run = start_run("train-estimate", root)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run.run_dir / "estimate.json",
                      json.dumps({"token_stats": stats.model_dump(), "estimate": est.model_dump()},
                                 ensure_ascii=False, indent=2) + "\n")  # fmt: skip
    atomic_write_text(run.run_dir / "estimate.md", render_estimate(cfg, est, stats))
    run.finish(config={"train": cfg.model_dump(mode="json"), "tps": args.tps},
               base_revision=cfg.model.revision,
               data_hashes={cfg.data.train_path.as_posix(): cfg.data.expected_sha256.train},
               metrics={"optimizer_steps": est.optimizer_steps,
                        "processed_tokens": est.processed_tokens,
                        "trainable_tokens": est.trainable_tokens, "gpu_hours": est.gpu_hours,
                        "cost": est.cost, "cost_ceiling": est.cost_ceiling,
                        "lora_trainable_params": est.lora_trainable_params},
               seed=cfg.seed, packages=("transformers", "tokenizers"))  # fmt: skip
    gpu_hours = "not measured (step 13)" if est.gpu_hours is None else f"{est.gpu_hours:.2f}"
    print(f"{stats.records} records, {est.optimizer_steps} optimizer steps, "
          f"{est.processed_tokens:,} tokens processed "
          f"({est.trainable_tokens:,} trainable)")  # fmt: skip
    print(f"gpu_hours={gpu_hours}, cost={est.cost}, ceiling by wall time={est.cost_ceiling} "
          f"{est.currency}; LoRA parameters {est.lora_trainable_params}")  # fmt: skip
    print(f"Report: {run.run_dir / 'estimate.md'}")
    return 0


def _resumed_config(checkpoint: Path, root: Path) -> TrainConfig:
    run_dir = checkpoint.parent
    try:
        stored = json.loads((run_dir / TRAIN_CONFIG_COPY).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QFError(f"{run_dir} has no {TRAIN_CONFIG_COPY}: not a training run") from exc
    return TrainConfig.model_validate(stored["train"])


@contextmanager
def _console_copy(path: Path) -> Iterator[None]:
    """stdout and stderr of this process also go to `path` (Python-level output: prints, progress
    bars, warnings; output written by C libraries directly is not captured)."""

    class Tee:
        def __init__(self, stream: TextIO, copy: TextIO) -> None:
            self.stream, self.copy = stream, copy

        def write(self, text: str) -> int:
            self.copy.write(text)
            return self.stream.write(text)

        def flush(self) -> None:
            self.copy.flush()
            self.stream.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.stream, name)

    with path.open("a", encoding="utf-8") as copy:
        saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(saved[0], copy), Tee(saved[1], copy)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved


def _resume_session(
    args: argparse.Namespace, root: Path
) -> tuple[TrainConfig, Path, Path, str | None, int]:
    if (
        args.config
        or args.experiment
        or args.seed is not None
        or any(v is not None for v in (args.max_steps, args.save_every, args.eval_every))
    ):
        raise QFError("--resume continues with the config the run was started with; do not "
                      "pass --config, --experiment, --seed or overrides")  # fmt: skip
    checkpoint = args.resume if args.resume.is_absolute() else root / args.resume
    run_dir = checkpoint.parent
    cfg = _resumed_config(checkpoint, root)
    previous = read_preflights(run_dir)
    experiment = previous[-1].experiment_id if previous else None
    return cfg, run_dir, checkpoint, experiment, len(previous) + 1


def run_run(args: argparse.Namespace) -> int:
    root = project_root()
    if args.resume is not None:
        cfg, run_dir, checkpoint, experiment, session = _resume_session(args, root)
    else:
        if args.allow_code_change is not None:
            raise QFError("--allow-code-change applies to --resume only")
        overrides = {"max_steps": args.max_steps,
                     "checkpoint_every_optimizer_steps": args.save_every,
                     "evaluate_every_optimizer_steps": args.eval_every}  # fmt: skip
        cfg, experiment = _selected_config(args, root, overrides)
        checkpoint, session = None, 1
        run_dir = root / cfg.output.run_root / new_run_id("train")
    paid = cfg.budget.gpu_hourly_rate is not None
    checks = ["data sha256 = config", "base model pinned", "tokenizer files verified",
              "max_steps within the planned epochs", "budget"]  # fmt: skip
    if paid and experiment is None:
        raise QFError("a paid run (budget.gpu_hourly_rate is set) must be a registered "
                      "experiment: --experiment EXXX (D-118)")  # fmt: skip
    git_commit, git_dirty = collect_git_info(root)
    spec = None
    if experiment is not None:
        spec = load_experiment(root, experiment)
        if git_commit is None or git_dirty:
            raise QFError(f"{experiment} needs a committed, clean git tree (commit {git_commit}, "
                          f"dirty {git_dirty}): the run must be traceable to its code")  # fmt: skip
        done = replicates_done(root / RUNS, experiment)
        if checkpoint is None and cfg.seed in done:
            raise QFError(f"{experiment} seed {cfg.seed} is already completed ({done[cfg.seed]})")
        checks += [f"experiment {experiment} registered, config hash matches", "git tree clean"]
    needs_approval = paid or (spec is not None and spec.requires_approval)
    approval = _approval(args.approval, "a paid run") if needs_approval else args.approval
    trainer, stats, est = _preflight(cfg, root, args.tps)
    append_preflight(run_dir, Preflight(
        session=session, created_at=datetime.now(UTC), run_id=run_dir.name,
        experiment_id=experiment, seed=cfg.seed, config_sha256=experiment_config_hash(cfg),
        approval=approval, allow_code_change=args.allow_code_change,
        resume_from=str(checkpoint.relative_to(root)) if checkpoint else None,
        git_commit=git_commit, git_dirty=git_dirty, host=machine_label(), checks=checks,
        token_stats=stats, estimate=est,
    ))  # fmt: skip
    print(f"{experiment or 'no experiment'}, seed {cfg.seed}: {est.optimizer_steps} optimizer "
          f"steps, {est.processed_tokens:,} tokens; wall-time limit "
          f"{cfg.training.max_wall_time_minutes} min; run {run_dir.name}")  # fmt: skip
    train_ref, val_ref = data_refs(cfg, root)
    with _console_copy(run_dir / CONSOLE_FILE):
        ref = trainer.train(cfg, train_ref, val_ref, run_dir, checkpoint)
    print(f"Adapter: {ref.path} (sha256 {ref.sha256[:12]}…)")
    return 0


def run_fetch_base(args: argparse.Namespace) -> int:
    root = project_root()
    _approval(args.approval, "downloading ~24.5 GB of base weights")
    cfg = load_yaml_config(root / args.config, BaseModelConfig)
    print(f"Downloading {len(cfg.weight_files)} files of {cfg.id}@{cfg.revision[:12]} "
          "(HF-format shards and index only)", flush=True)  # fmt: skip
    directory = download_base_weights(cfg, root)
    provenance = verify_base_weights(cfg, root)
    total = sum(info["bytes"] for info in provenance["files"].values())
    print(f"Base model weights, {total / 1e9:.1f} GB, sha256 checked against the Hub: "
          f"{directory}")  # fmt: skip
    return 0


def run_verify(args: argparse.Namespace) -> int:
    """A training run as copied (e.g. from the GPU machine): check it before deleting the source."""
    root = project_root()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    check = verify_run(run_dir, root)
    for line in check.checks:
        print(f"  ok: {line}")
    for line in check.problems:
        print(f"  PROBLEM: {line}")
    print(f"{run_dir.name}: {'intact' if check.ok else 'NOT intact'} ({check.parameters:,} LoRA "
          f"parameters in {check.tensors} tensors)")  # fmt: skip
    return 0 if check.ok else 1
