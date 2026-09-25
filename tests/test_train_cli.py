"""`qf train estimate` and `qf train run` (plan step 12): the refusals before any model is
loaded, "CUDA not available" on a machine without a GPU, and a full CPU run with resume."""

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
from qf.training.trainers.hf_qlora import callbacks as callbacks_module  # noqa: E402
from tests.test_train_tiny import jumping_clock  # noqa: E402
from tests.tiny_training import TinyProject, tiny_config, tiny_project  # noqa: E402


@pytest.fixture
def project(fake_project: Path) -> TinyProject:
    return tiny_project(fake_project)


def write_config(project: TinyProject, cfg: TrainConfig, name: str = "tiny.yaml") -> str:
    path = project.root / "configs/train" / name
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True), "utf-8")
    return str(path.relative_to(project.root))


def run(*args: str) -> int:
    return main(["train", *args])


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


@pytest.mark.skipif(torch.cuda.is_available(), reason="the message of a machine without CUDA")
def test_run_without_gpu_says_cuda_not_available(
    project: TinyProject, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tiny_config(project, trainer={"loader": "cuda_4bit"})
    cfg = cfg.model_copy(update={"quantization": cfg.quantization.model_copy(
        update={"load_in_4bit": True})})  # fmt: skip
    assert run("run", "--config", write_config(project, cfg)) == 1
    assert "CUDA not available" in capsys.readouterr().err


@pytest.mark.parametrize(("changes", "message"), [
    ({"budget": {"gpu_hourly_rate": 231.17, "max_cost": 100.0}}, "over budget"),
    ({"max_steps": 9}, "max_steps 9 exceeds 8"),
])  # fmt: skip
def test_run_refuses_before_loading_a_model(
    project: TinyProject, changes: dict[str, Any], message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    assert run("run", "--config", write_config(project, tiny_config(project, **changes))) == 1
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
def test_run_stop_and_resume(project: TinyProject, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    config = write_config(project, tiny_config(project, max_steps=6))
    with monkeypatch.context() as patched:
        patched.setattr(callbacks_module, "_now", jumping_clock(3))
        assert run("run", "--config", config) == 1
    err = capsys.readouterr().err
    assert "training aborted at step 3: wall_time" in err and "--resume runs/" in err
    run_dir = next((project.root / "runs").glob("*-train-*"))
    assert run("run", "--resume", str((run_dir / "checkpoint-3").relative_to(project.root))) == 0
    assert "Adapter: runs/" in capsys.readouterr().out
    manifest = read_manifest(run_dir)
    assert manifest.status == "completed" and manifest.metrics["global_step"] == 6
