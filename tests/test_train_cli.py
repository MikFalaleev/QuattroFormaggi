"""`qf train estimate|run|verify` (plan step 12, experiment protocol D-118): the refusals before
any model is loaded, "CUDA not available" on a machine without a GPU, and full CPU runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

import torch  # noqa: E402

from qf.cli.main import main  # noqa: E402
from qf.common import read_manifest  # noqa: E402
from qf.contracts import TrainConfig  # noqa: E402
from qf.training import read_events, read_preflights  # noqa: E402
from qf.training.trainers.hf_qlora import callbacks as callbacks_module  # noqa: E402
from tests.test_train_tiny import jumping_clock  # noqa: E402
from tests.tiny_training import (  # noqa: E402
    TinyProject,
    git_init,
    register,
    tiny_config,
    tiny_project,
)

APPROVAL = "user, 2026-09-25: «Да, разрешаю»"
PAID = {"gpu_hourly_rate": 1.0, "max_cost": 100.0}  # a paid run that fits the budget


@pytest.fixture
def project(fake_project: Path) -> TinyProject:
    return tiny_project(fake_project)


def write_config(project: TinyProject, cfg: TrainConfig, name: str = "tiny.yaml") -> str:
    path = project.root / "configs/train" / name
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True), "utf-8")
    return str(path.relative_to(project.root))


def run(*args: str) -> int:
    return main(["train", *args])


def committed(project: TinyProject, cfg: TrainConfig, **kwargs: Any) -> str:
    """A registered experiment of `cfg` in a committed, clean git tree."""
    experiment = register(project, cfg, **kwargs)
    git_init(project.root)
    return experiment


def test_estimate_writes_a_report(project: TinyProject, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(project, tiny_config(project, budget={"gpu_hourly_rate": 231.17}))
    assert run("estimate", "--config", config) == 0
    out = capsys.readouterr().out
    assert "gpu_hours=not measured (step 13)" in out
    report_dir = next((project.root / "runs").glob("*-train-estimate-*"))
    report = (report_dir / "estimate.md").read_text("utf-8")
    assert "не измерена" in report and "предположения, не замер" in report
    estimate = json.loads((report_dir / "estimate.json").read_text("utf-8"))["estimate"]
    assert estimate["gpu_hours"] is None and estimate["cost_ceiling"] == pytest.approx(231.17)
    assert run("estimate", "--config", config, "--tps", "100") == 0
    assert "gpu_hours=0." in capsys.readouterr().out


def test_estimate_of_a_registered_experiment(project: TinyProject) -> None:
    experiment = committed(project, tiny_config(project, budget=PAID))
    assert run("estimate", "--experiment", experiment) == 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="the message of a machine without CUDA")
def test_run_without_gpu_says_cuda_not_available(
    project: TinyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tiny_config(project, trainer={"loader": "cuda_4bit"}, budget=PAID)
    cfg = cfg.model_copy(update={"quantization": cfg.quantization.model_copy(
        update={"load_in_4bit": True})})  # fmt: skip
    experiment = committed(project, cfg)
    assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    assert "CUDA not available" in capsys.readouterr().err
    run_dir = next((project.root / "runs").glob("*-train-*"))
    assert read_manifest(run_dir).status == "failed"  # recorded, not lost
    assert read_preflights(run_dir)[0].approval == APPROVAL
    assert [e["event"] for e in read_events(run_dir)] == ["started", "failed"]


@pytest.mark.parametrize(("args", "message"), [
    (["--config", "configs/train/paid.yaml"], "must be a registered experiment"),
    (["--experiment", "E901", "--max-steps", "3"], "no --config or overrides"),
    (["--experiment", "E901", "--config", "configs/train/paid.yaml"], "no --config or overrides"),
    (["--experiment", "E901", "--seed", "8"], "seed 8 is not registered"),
    (["--experiment", "E901"], "needs the human decision"),
    (["--experiment", "E901", "--approval", "ok"], "needs the human decision"),
    (["--experiment", "E999", "--approval", APPROVAL], "E999 is not registered"),
    (["--config", "configs/train/paid.yaml", "--seed", "7"], "use it with --experiment"),
    (["--experiment", "E901", "--approval", APPROVAL, "--allow-code-change", "x"],
     "applies to --resume only"),
])  # fmt: skip
def test_protocol_refusals(project: TinyProject, capsys: pytest.CaptureFixture[str],
                           args: list[str], message: str) -> None:  # fmt: skip
    write_config(project, tiny_config(project, budget=PAID), "paid.yaml")
    committed(project, tiny_config(project, budget=PAID))
    assert run("run", *args) == 1
    assert message in capsys.readouterr().err
    assert not list(project.root.glob("runs/*-train-*"))  # nothing started


def test_experiment_config_changed_after_registration(
    project: TinyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = committed(project, tiny_config(project, budget=PAID))
    write_config(project, tiny_config(project, budget=PAID, learning_rate=1e-3), "e901.yaml")
    assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    assert "changed since E901 was registered" in capsys.readouterr().err


def test_experiment_needs_a_clean_tree(project: TinyProject,
                                       capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    experiment = committed(project, tiny_config(project, budget=PAID))
    (project.root / "configs/train/extra.yaml").write_text("x: 1\n", "utf-8")  # uncommitted
    assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    assert "needs a committed, clean git tree" in capsys.readouterr().err


@pytest.mark.parametrize(("changes", "message"), [
    ({"budget": {"gpu_hourly_rate": 231.17, "max_cost": 100.0}}, "over budget"),
    ({"max_steps": 9, "budget": PAID}, "max_steps 9 exceeds 8"),
])  # fmt: skip
def test_run_refuses_before_loading_a_model(
    project: TinyProject, changes: dict[str, Any], message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    experiment = committed(project, tiny_config(project, **changes))
    assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    assert message in capsys.readouterr().err
    assert not list(project.root.glob("runs/*-train-*"))  # no run directory was created


def test_run_refuses_changed_data(project: TinyProject, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(project, tiny_config(project))
    (project.root / project.val_path).write_text("changed\n", encoding="utf-8")
    assert run("run", "--config", config) == 1
    assert "content hash mismatch" in capsys.readouterr().err


def test_resume_takes_no_config(project: TinyProject, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("run", "--resume", "runs/x/checkpoint-1", "--max-steps", "3") == 1
    assert "--resume continues with the config" in capsys.readouterr().err


@pytest.mark.slow
def test_registered_run_records_the_protocol(
    project: TinyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = committed(project, tiny_config(project, budget=PAID, max_steps=4),
                           seeds=[7, 8])  # fmt: skip
    assert run("run", "--experiment", experiment, "--seed", "8", "--approval", APPROVAL) == 0
    run_dir = next((project.root / "runs").glob("*-train-*"))
    manifest = read_manifest(run_dir)
    assert manifest.status == "completed" and manifest.experiment_id == "E901"
    assert manifest.seed == 8 and manifest.git_dirty is False
    preflight = read_preflights(run_dir)[0]
    assert preflight.approval == APPROVAL and preflight.estimate.optimizer_steps == 4
    assert "git tree clean" in preflight.checks
    events = [e["event"] for e in read_events(run_dir)]
    assert events[:2] == ["started", "model_loaded"] and events[-2:] == ["completed", "samples"]
    assert {"checkpoint", "evaluated", "integrity"} <= set(events)  # samples after completion
    assert (run_dir / "resources.jsonl").read_text("utf-8").count("\n") >= 2
    assert manifest.metrics["rss_peak_bytes"] and manifest.metrics["integrity_ok"]
    assert "Образцы ответов — не оценка" in (run_dir / "samples.md").read_text("utf-8")
    assert "trainable parameters" in (run_dir / "console.log").read_text("utf-8")
    assert json.loads((run_dir / "integrity.json").read_text("utf-8"))["ok"]
    capsys.readouterr()
    assert run("verify", str(run_dir.relative_to(project.root))) == 0
    assert "intact" in capsys.readouterr().out
    # a completed replicate is not run again; another registered seed is
    assert run("run", "--experiment", experiment, "--seed", "8", "--approval", APPROVAL) == 1
    assert "seed 8 is already completed" in capsys.readouterr().err
    # a damaged copy is found
    weights = run_dir / "adapter" / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-8] + b"\x00" * 8)
    assert run("verify", str(run_dir.relative_to(project.root))) == 1
    assert "content hash mismatch" in capsys.readouterr().out


@pytest.mark.slow
def test_run_stop_and_resume(project: TinyProject, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    experiment = committed(project, tiny_config(project, budget=PAID, max_steps=6))
    with monkeypatch.context() as patched:
        patched.setattr(callbacks_module, "_now", jumping_clock(3))
        assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    err = capsys.readouterr().err
    assert "training aborted at step 3: wall_time" in err and "--resume runs/" in err
    run_dir = next((project.root / "runs").glob("*-train-*"))
    checkpoint = str((run_dir / "checkpoint-3").relative_to(project.root))
    assert run("run", "--resume", checkpoint) == 1  # a paid resume needs its own approval
    assert "needs the human decision" in capsys.readouterr().err
    assert run("run", "--resume", checkpoint, "--approval", APPROVAL) == 0
    assert "Adapter: runs/" in capsys.readouterr().out
    manifest = read_manifest(run_dir)
    assert manifest.status == "completed" and manifest.metrics["global_step"] == 6
    assert [p.session for p in read_preflights(run_dir)] == [1, 2]
    assert "resumed" in [e["event"] for e in read_events(run_dir)]


@pytest.mark.slow
def test_resume_refuses_other_code_unless_accepted(
    project: TinyProject, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    experiment = committed(project, tiny_config(project, budget=PAID, max_steps=6))
    with monkeypatch.context() as patched:
        patched.setattr(callbacks_module, "_now", jumping_clock(3))
        assert run("run", "--experiment", experiment, "--approval", APPROVAL) == 1
    (project.root / "NOTES.md").write_text("a code change\n", "utf-8")
    subprocess.run(["git", "add", "-A"], cwd=project.root, check=True)
    subprocess.run(["git", "-c", "user.email=t@e.com", "-c", "user.name=t", "commit", "-qm", "x"],
                   cwd=project.root, check=True)  # fmt: skip
    run_dir = next((project.root / "runs").glob("*-train-*"))
    checkpoint = str((run_dir / "checkpoint-3").relative_to(project.root))
    capsys.readouterr()
    assert run("run", "--resume", checkpoint, "--approval", APPROVAL) == 1
    assert "code commit" in capsys.readouterr().err
    assert run("run", "--resume", checkpoint, "--approval", APPROVAL,
               "--allow-code-change", "notes only, no code") == 0  # fmt: skip
    resumed = [e for e in read_events(run_dir) if e["event"] == "resumed"]
    assert resumed[-1]["allow_code_change"] == "notes only, no code"
