"""Checkpoint/resume equivalence on CPU (plan step 12).

Run A trains 8 steps without a break. Run B is stopped by the wall-time limit after step 4
(checkpoint-4), then a new Trainer resumes it to step 8. The LoRA weights, the losses of steps
5–8 and the records of every step must match: optimizer, scheduler, RNG (dropout) and data
position are all restored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from qf.common import read_manifest  # noqa: E402
from qf.training import data_refs  # noqa: E402
from qf.training.trainers.hf_qlora import TrainingStopped, run_training  # noqa: E402
from qf.training.trainers.hf_qlora import callbacks as callbacks_module  # noqa: E402
from qf.training.trainers.hf_qlora.config import HFQLoRATrainerConfig  # noqa: E402
from tests.test_train_tiny import jumping_clock  # noqa: E402
from tests.tiny_training import tiny_config, tiny_project, train_log  # noqa: E402

pytestmark = pytest.mark.slow
TINY = HFQLoRATrainerConfig(loader="tiny_random_cpu")


def steps(run_dir: Path) -> dict[int, dict[str, object]]:
    return {line["step"]: line for line in train_log(run_dir) if "loss" in line}


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_resume_equivalence(fake_project: Path, monkeypatch: pytest.MonkeyPatch,
                            dropout: float) -> None:  # fmt: skip
    project = tiny_project(fake_project)
    cfg = tiny_config(project, dropout=dropout, max_steps=8)
    train_ref, val_ref = data_refs(cfg, project.root)
    run_a, run_b = project.root / "runs/a", project.root / "runs/b"
    run_training(cfg, TINY, train_ref, val_ref, run_a, None, root=project.root)

    with monkeypatch.context() as patched, pytest.raises(TrainingStopped, match="wall_time"):
        patched.setattr(callbacks_module, "_now", jumping_clock(4))
        run_training(cfg, TINY, train_ref, val_ref, run_b, None, root=project.root)
    assert sorted(p.name for p in run_b.glob("checkpoint-*")) == ["checkpoint-4"]
    run_training(cfg, TINY, train_ref, val_ref, run_b, run_b / "checkpoint-4", root=project.root)

    a, b = steps(run_a), steps(run_b)
    assert sorted(a) == sorted(b) == list(range(1, 9))
    for step in range(1, 9):  # the same records in the same order, before and after resume
        assert a[step]["record_ids"] == b[step]["record_ids"], step
    for step in range(5, 9):
        assert a[step]["loss"] == pytest.approx(b[step]["loss"], abs=1e-6), step
        assert a[step]["grad_norm"] == pytest.approx(b[step]["grad_norm"], abs=1e-6), step
        assert a[step]["learning_rate"] == b[step]["learning_rate"], step
    weights_a = load_file(run_a / "adapter/adapter_model.safetensors")
    weights_b = load_file(run_b / "adapter/adapter_model.safetensors")
    assert weights_a.keys() == weights_b.keys()
    for name in weights_a:
        assert torch.allclose(weights_a[name], weights_b[name], atol=1e-6), name
    manifest = read_manifest(run_b)
    assert manifest.status == "completed"
    assert manifest.parent_checkpoint == "runs/b/checkpoint-4"
    assert manifest.metrics["resumed_from"] == ["runs/b/checkpoint-4"]
