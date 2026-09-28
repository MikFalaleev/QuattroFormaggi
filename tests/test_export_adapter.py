"""Step 15 (D-121): the backup copy of an accepted adapter and its check against the manifests,
on fake adapter files (no model, no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qf.cli.main import main
from qf.common import QFError, load_artifact_manifest, write_artifact
from qf.contracts import ModelRef
from qf.export import ADAPTERS_DIR, backup_adapter, verify_adapter

BASE = "mistralai/Mistral-Nemo-Instruct-2407"
REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
RUN = "20260101-000000-train-abc123"
SOURCE = Path(f"runs/{RUN}/adapter")
PINNED = ModelRef(id=BASE, revision=REVISION)


def fake_adapter(root: Path, **own: Any) -> Path:
    """An adapter directory as `qf train run` leaves it, with its artifact manifest."""
    directory = root / SOURCE
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"\x00fake lora weights\x01" * 64)
    config = {"base_model_name_or_path": BASE, "revision": REVISION, "r": 16}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    manifest = {"base_model": BASE, "revision": REVISION, "loader": "cuda_4bit", "run_id": RUN,
                "experiment_id": "E302", "seed": 42, "global_step": 94,
                "quantization": "nf4, double quant True (bitsandbytes)", **own}  # fmt: skip
    (directory / "qf_adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_artifact(directory, "lora_adapter", "peft_lora_v1", RUN, ["a" * 64],
                   extra={"loader": manifest["loader"]}, root=root)  # fmt: skip
    return SOURCE


def pinned_base_config(root: Path) -> None:
    path = root / "configs/train/base_model.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"id: {BASE}\nrevision: {REVISION}\nlicense: apache-2.0\n"
                    "tokenizer_files: [tokenizer.json]\nlocal_dir: artifacts/base_model/x\n",
                    encoding="utf-8")  # fmt: skip


def test_adapter_manifest_hash_check(fake_project: Path) -> None:
    fake_adapter(fake_project)
    check = verify_adapter(SOURCE, fake_project, base=PINNED)
    assert check.ok, check.problems
    assert any("matches its manifest" in line for line in check.checks)
    weights = fake_project / SOURCE / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")  # one changed byte
    check = verify_adapter(SOURCE, fake_project, base=PINNED)
    assert not check.ok and "content hash mismatch" in check.problems[0]


def test_backup_is_an_identical_copy_with_its_own_manifest(fake_project: Path) -> None:
    fake_adapter(fake_project)
    ref = backup_adapter(SOURCE, "qf-12b-v0.1-seed42", fake_project)
    source_manifest = load_artifact_manifest(fake_project / SOURCE, root=fake_project)
    assert ref.path.as_posix() == f"{ADAPTERS_DIR.as_posix()}/qf-12b-v0.1-seed42"
    assert ref.sha256 == source_manifest.ref.sha256
    assert ref.producer_run_id == RUN and ref.parents == source_manifest.ref.parents
    extra = load_artifact_manifest(fake_project / ref.path, root=fake_project).extra
    assert extra["backup_of"] == SOURCE.as_posix() and extra["loader"] == "cuda_4bit"
    check = verify_adapter(Path(ref.path), fake_project, base=PINNED)
    assert check.ok and any("the same sha256" in line for line in check.checks)


def test_backup_refuses_an_existing_name_and_a_bad_name(fake_project: Path) -> None:
    fake_adapter(fake_project)
    backup_adapter(SOURCE, "seed42", fake_project)
    with pytest.raises(QFError, match="already exists"):
        backup_adapter(SOURCE, "seed42", fake_project)
    with pytest.raises(QFError, match="backup name"):
        backup_adapter(SOURCE, "../outside", fake_project)


def test_backup_refuses_a_source_that_does_not_match_its_manifest(fake_project: Path) -> None:
    fake_adapter(fake_project)
    (fake_project / SOURCE / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        backup_adapter(SOURCE, "seed42", fake_project)
    assert not (fake_project / ADAPTERS_DIR / "seed42").exists()


def test_backup_reports_a_source_that_changed_later(fake_project: Path) -> None:
    fake_adapter(fake_project)
    ref = backup_adapter(SOURCE, "seed42", fake_project)
    (fake_project / SOURCE / "adapter_config.json").write_text("{}", encoding="utf-8")
    check = verify_adapter(Path(ref.path), fake_project, base=PINNED)
    assert not check.ok and check.problems[0].startswith(f"source {SOURCE.as_posix()}")


def test_backup_without_its_source_on_this_machine_is_fine(fake_project: Path) -> None:
    fake_adapter(fake_project)
    ref = backup_adapter(SOURCE, "seed42", fake_project)
    for path in (fake_project / SOURCE).iterdir():
        path.unlink()
    (fake_project / SOURCE).rmdir()
    check = verify_adapter(Path(ref.path), fake_project, base=PINNED)
    assert check.ok and any("not on this machine" in line for line in check.checks)


@pytest.mark.parametrize(("own", "problem"), [
    ({"loader": "tiny_random_cpu"}, "loader 'tiny_random_cpu'"),
    ({"revision": "b" * 40}, "adapter_config names"),
    ({"run_id": "20260101-000000-train-other0"}, "names run"),
])  # fmt: skip
def test_adapter_problems_are_reported(fake_project: Path, own: dict[str, Any],
                                       problem: str) -> None:  # fmt: skip
    fake_adapter(fake_project, **own)
    check = verify_adapter(SOURCE, fake_project, base=PINNED)
    assert not check.ok and any(problem in p for p in check.problems)


def test_adapter_of_another_base_is_reported(fake_project: Path) -> None:
    fake_adapter(fake_project)
    other = ModelRef(id="other/model", revision="c" * 40)
    check = verify_adapter(SOURCE, fake_project, base=other)
    assert any("not the pinned other/model" in p for p in check.problems)


def test_missing_files_are_reported(fake_project: Path) -> None:
    fake_adapter(fake_project)
    (fake_project / SOURCE / "adapter_config.json").unlink()
    write_artifact(fake_project / SOURCE, "lora_adapter", "peft_lora_v1", RUN, root=fake_project)
    check = verify_adapter(SOURCE, fake_project)
    assert check.problems == ["missing files: adapter_config.json"]


def test_cli_backup_and_verify(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake_adapter(fake_project)
    pinned_base_config(fake_project)
    assert main(["export", "adapter", f"runs/{RUN}", "--name", "seed42"]) == 0
    out = capsys.readouterr().out
    assert "Backup: artifacts/adapters/seed42" in out and "⛔" in out
    assert main(["export", "verify", "--adapter", "artifacts/adapters/seed42"]) == 0
    assert "artifacts/adapters/seed42: intact" in capsys.readouterr().out
    weights = fake_project / ADAPTERS_DIR / "seed42" / "adapter_model.safetensors"
    weights.write_bytes(b"broken")
    assert main(["export", "verify", "--adapter", "artifacts/adapters/seed42"]) == 1
    assert "NOT intact" in capsys.readouterr().out


def test_cli_does_not_copy_a_broken_source(fake_project: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    fake_adapter(fake_project, loader="tiny_random_cpu")
    pinned_base_config(fake_project)
    assert main(["export", "adapter", f"runs/{RUN}", "--name", "seed42"]) == 1
    assert "not copied" in capsys.readouterr().out
    assert not (fake_project / ADAPTERS_DIR / "seed42").exists()
