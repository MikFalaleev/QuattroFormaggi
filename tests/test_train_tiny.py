"""The training pipeline on CPU with a tiny random Mistral and a fake tokenizer (plan step 12).

Nothing is downloaded. Marked `slow`: run with `uv run pytest -m slow tests/test_train_tiny.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

import torch  # noqa: E402

from qf.common import read_artifact, read_manifest  # noqa: E402
from qf.contracts import TrainConfig  # noqa: E402
from qf.training import data_refs  # noqa: E402
from qf.training.trainers.hf_qlora import TrainingStopped, run_training  # noqa: E402
from qf.training.trainers.hf_qlora import callbacks as callbacks_module  # noqa: E402
from qf.training.trainers.hf_qlora import train as train_module  # noqa: E402
from qf.training.trainers.hf_qlora.config import HFQLoRATrainerConfig  # noqa: E402
from tests.tiny_training import (  # noqa: E402
    TinyProject,
    git_init,
    tiny_config,
    tiny_project,
    train_log,
)

pytestmark = pytest.mark.slow
TINY = HFQLoRATrainerConfig(loader="tiny_random_cpu")


@pytest.fixture
def project(fake_project: Path) -> TinyProject:
    return tiny_project(fake_project)


def train(project: TinyProject, cfg: TrainConfig, name: str = "run",
          resume: Path | None = None) -> Path:  # fmt: skip
    train_ref, val_ref = data_refs(cfg, project.root)
    run_dir = project.root / "runs" / name
    run_training(cfg, TINY, train_ref, val_ref, run_dir, resume, root=project.root)
    return run_dir


def losses(run_dir: Path) -> dict[int, float]:
    return {line["step"]: line["loss"] for line in train_log(run_dir) if "loss" in line}


def test_loss_decreases_on_repeated_batch(project: TinyProject) -> None:
    """30 steps, each over all 16 records in one batch: the loss falls by more than half."""
    cfg = tiny_config(project, micro_batch_size=16, epochs=30, max_steps=30, learning_rate=1e-2)
    curve = losses(train(project, cfg))
    assert len(curve) == 30 and curve[30] < 0.5 * curve[1], curve


def test_only_lora_params_change(project: TinyProject, monkeypatch: pytest.MonkeyPatch) -> None:
    def snapshot(model: Any, lora: bool) -> dict[str, str]:
        return {name: hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
                for name, p in model.named_parameters() if ("lora_" in name) == lora}  # fmt: skip

    original_apply = train_module.apply_lora
    seen: dict[str, Any] = {}

    def apply_and_remember(model: Any, *args: Any, **kwargs: Any) -> Any:
        peft_model = original_apply(model, *args, **kwargs)
        seen.update(model=peft_model, base=snapshot(peft_model, False),
                    lora=snapshot(peft_model, True))  # fmt: skip
        return peft_model

    monkeypatch.setattr(train_module, "apply_lora", apply_and_remember)
    train(project, tiny_config(project, max_steps=4))
    after_base, after_lora = snapshot(seen["model"], False), snapshot(seen["model"], True)
    assert after_base == seen["base"] and len(after_base) > 20
    changed = [name for name in after_lora if after_lora[name] != seen["lora"][name]]
    assert changed and any("lora_A" in n for n in changed) and any("lora_B" in n for n in changed)


def test_final_eval_and_save_even_if_interval_not_reached(project: TinyProject) -> None:
    cfg = tiny_config(project, max_steps=3, checkpoint_every_optimizer_steps=50,
                      evaluate_every_optimizer_steps=50)  # fmt: skip
    run_dir = train(project, cfg)
    ref = read_artifact(run_dir / "adapter", "lora_adapter", {"peft_lora_v1"}, root=project.root)
    assert (run_dir / "adapter" / "adapter_model.safetensors").exists()
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    assert not (run_dir / "adapter" / "tokenizer.json").exists()  # a reference, not a copy
    assert ref.parents == (cfg.data.expected_sha256.train, cfg.data.expected_sha256.val)
    manifest = read_manifest(run_dir)
    assert manifest.status == "completed" and manifest.metrics["global_step"] == 3
    assert math.isfinite(manifest.metrics["final_eval_loss"])
    assert any("eval_loss" in line and line["step"] == 3 for line in train_log(run_dir))
    assert (run_dir / "checkpoint-3").is_dir()


def test_adapter_of_the_test_model_does_not_claim_a_base(project: TinyProject) -> None:
    run_dir = train(project, tiny_config(project, max_steps=1))
    adapter_config = json.loads((run_dir / "adapter/adapter_config.json").read_text("utf-8"))
    assert adapter_config["base_model_name_or_path"] == train_module.TEST_MODEL_BASE
    assert adapter_config["revision"] is None
    assert not (run_dir / "adapter/README.md").exists()
    qf_manifest = json.loads((run_dir / "adapter/qf_adapter_manifest.json").read_text("utf-8"))
    assert qf_manifest["loader"] == "tiny_random_cpu" and qf_manifest["chat_template_hash"]


def test_adapter_names_the_pinned_hub_model(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/home/u/artifacts/base", "revision": None})
    )
    (tmp_path / "README.md").write_text("base_model: /home/u/artifacts/base")
    train_module.name_base_model(tmp_path, "mistralai/Mistral-Nemo-Instruct-2407", "b" * 40)
    config = json.loads((tmp_path / "adapter_config.json").read_text("utf-8"))
    assert config["base_model_name_or_path"] == "mistralai/Mistral-Nemo-Instruct-2407"
    assert config["revision"] == "b" * 40 and not (tmp_path / "README.md").exists()


def jumping_clock(stop_after_steps: int) -> Any:
    """A clock for WallTimeLimitCallback: calls 1–2 are __init__ and on_train_begin, then one
    call per optimizer step; after `stop_after_steps` steps the limit is exceeded."""
    calls = {"n": 0}

    def now() -> float:
        calls["n"] += 1
        return 1e9 if calls["n"] > 2 + stop_after_steps - 1 else 0.0

    return now


def test_wall_time_callback_stops(project: TinyProject, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(callbacks_module, "_now", jumping_clock(2))
    with pytest.raises(TrainingStopped, match="wall_time"):
        train(project, tiny_config(project, max_steps=8))
    run_dir = project.root / "runs/run"
    manifest = read_manifest(run_dir)
    assert manifest.status == "aborted" and manifest.metrics["stop_reason"] == "wall_time"
    assert manifest.metrics["last_checkpoint"] == "runs/run/checkpoint-2"
    assert (run_dir / "checkpoint-2").is_dir() and not (run_dir / "adapter").exists()


def test_nan_guard_stops(project: TinyProject, monkeypatch: pytest.MonkeyPatch) -> None:
    original = train_module.QFTrainer.compute_loss

    def nan_loss(self: Any, model: Any, inputs: Any, return_outputs: bool = False,
                 num_items_in_batch: Any = None) -> Any:  # fmt: skip
        result = original(self, model, inputs, return_outputs, num_items_in_batch)
        if return_outputs:
            return result[0] * float("nan"), result[1]
        return result * float("nan")

    monkeypatch.setattr(train_module.QFTrainer, "compute_loss", nan_loss)
    with pytest.raises(TrainingStopped, match="failed"):
        train(project, tiny_config(project, max_steps=8))
    run_dir = project.root / "runs/run"
    manifest = read_manifest(run_dir)
    assert manifest.status == "failed" and "loss is nan" in manifest.metrics["stop_reason"]
    assert (run_dir / "checkpoint-1").is_dir()
    assert max(losses(run_dir)) == 1  # stopped at the first step


def test_termination_signal_is_recorded_as_aborted(
    project: TinyProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `kill` (SIGTERM) or a dropped SSH session (SIGHUP) during training: status aborted,
    the last checkpoint named, and the signal handlers restored afterwards."""
    import os
    import signal

    original = train_module.QFTrainer.compute_loss
    calls = {"n": 0}

    def terminated_at_step_3(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        if model.training:
            calls["n"] += 1
            if calls["n"] == 5:  # micro-batch 5 = the first of step 3 (2 per step)
                os.kill(os.getpid(), signal.SIGTERM)
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(train_module.QFTrainer, "compute_loss", terminated_at_step_3)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(TrainingStopped, match=r"aborted at step 2: interrupted \(SIGTERM\)"):
        cfg = tiny_config(project, max_steps=8, gradient_accumulation_steps=2,
                          micro_batch_size=1, checkpoint_every_optimizer_steps=1)  # fmt: skip
        train(project, cfg)
    assert signal.getsignal(signal.SIGTERM) is before
    manifest = read_manifest(project.root / "runs/run")
    assert manifest.status == "aborted" and manifest.metrics["last_checkpoint"].endswith("-2")


def test_manifest_complete(project: TinyProject) -> None:
    git_init(project.root)
    run_dir = train(project, tiny_config(project, max_steps=2))
    manifest = read_manifest(run_dir)
    assert manifest.git_commit and manifest.git_dirty is False
    assert manifest.base_revision == "a" * 40 and manifest.seed == 7
    assert set(manifest.data_hashes) == {"data/train.jsonl", "data/val.jsonl"}
    assert manifest.tokenizer_hash and manifest.chat_template_hash
    assert set(manifest.prompt_hashes) == {"system"}
    assert manifest.config["train"]["training"]["max_steps"] == 2
    assert manifest.package_versions["torch"] and manifest.package_versions["peft"]
    assert manifest.hardware["cpu_count_logical"] and manifest.wall_time_s
    assert manifest.metrics["trainable_params"] > 0 and manifest.metrics["session_tokens"] > 0
    assert manifest.peak_memory_bytes is None  # CPU: GPU memory is not measured
    assert manifest.parent_checkpoint is None


def test_exactly_one_epoch_sees_every_record_once(fake_project: Path) -> None:
    """10 records, micro 1 × accum 4: HF Trainer's steps equal ceil(10 / 4) = 3 and the last
    step is shorter; every record is trained on exactly once."""
    project = tiny_project(fake_project, n_train=10)
    cfg = tiny_config(project, micro_batch_size=1, gradient_accumulation_steps=4, max_steps=3)
    run_dir = train(project, cfg)
    seen = [rid for line in train_log(run_dir) if "loss" in line for rid in line["record_ids"]]
    assert Counter(seen) == Counter(f"qf-train-LOAD{i:05d}-0" for i in range(10))
    assert [len(line["record_ids"]) for line in train_log(run_dir) if "loss" in line] == [4, 4, 2]


def test_resume_needs_the_same_config(project: TinyProject,
                                      monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    with monkeypatch.context() as patched, pytest.raises(TrainingStopped):
        patched.setattr(callbacks_module, "_now", jumping_clock(2))
        train(project, tiny_config(project, max_steps=8))
    other = tiny_config(project, max_steps=8, learning_rate=1e-3)
    with pytest.raises(Exception, match="config differs"):
        train(project, other, resume=project.root / "runs/run/checkpoint-2")


def test_tokens_are_counted(project: TinyProject) -> None:
    run_dir = train(project, tiny_config(project, max_steps=8))  # one epoch of 16 records
    manifest = read_manifest(run_dir)
    assert manifest.metrics["session_tokens"] == manifest.metrics["train_tokens"]


def test_cpu_loader_never_uses_mps(project: TinyProject) -> None:
    cfg = tiny_config(project, max_steps=1)
    train(project, cfg)
    args = torch.load(project.root / "runs/run/adapter/training_args.bin", weights_only=False)
    assert args.use_cpu is True
