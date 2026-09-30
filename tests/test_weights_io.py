"""Fetching and verifying the base model weights with a fake Hub (plan step 12; the real
download is a ⛔ STOP of step 13)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from qf.cli.main import main
from qf.common import QFError, load_yaml_config
from qf.training import BaseModelConfig, download_base_weights, verify_base_weights
from qf.training import weights_io as weights_module

FILES = {"model.safetensors.index.json": b'{"weight_map": {}}',
         "model-00001-of-00001.safetensors": b"weights" * 100}  # fmt: skip


def base_config(root: Path, weight_files: list[str] | None = None) -> BaseModelConfig:
    names = weight_files if weight_files is not None else list(FILES)
    path = root / "configs/train/base_model.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"id: fake/model\nrevision: {'a' * 40}\nlicense: apache-2.0\n"
                    "tokenizer_files: [tokenizer.json]\nlocal_dir: artifacts/base_model/fake\n"
                    f"weight_files: [{', '.join(names)}]\n", encoding="utf-8")  # fmt: skip
    return load_yaml_config(path, BaseModelConfig)


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    served = dict(FILES)
    listed = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()
                     if name.endswith(".safetensors") else None}
              for name, data in FILES.items()}  # fmt: skip
    listed["consolidated.safetensors"] = {"bytes": 10**10, "sha256": "f" * 64}

    def download(repo_id: str, filename: str, revision: str, local_dir: Path) -> str:
        assert filename != "consolidated.safetensors"
        target = Path(local_dir) / filename
        target.write_bytes(served[filename])
        return str(target)

    monkeypatch.setattr(weights_module, "hub_files", lambda cfg: listed)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return served


def test_fetch_and_verify(fake_project: Path, fake_hub: dict[str, bytes]) -> None:
    cfg = base_config(fake_project)
    directory = download_base_weights(cfg, fake_project)
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [*FILES, "weights_provenance.json"]
    )
    provenance = verify_base_weights(cfg, fake_project, full_hash=True)
    assert provenance["revision"] == "a" * 40 and set(provenance["files"]) == set(FILES)


def test_corrupted_download_is_refused(fake_project: Path, fake_hub: dict[str, bytes]) -> None:
    fake_hub["model-00001-of-00001.safetensors"] = b"weightz" * 100  # same size, other bytes
    with pytest.raises(QFError, match="the Hub reports"):
        download_base_weights(base_config(fake_project), fake_project)


def test_changed_or_missing_weights_are_found(fake_project: Path,
                                              fake_hub: dict[str, bytes]) -> None:  # fmt: skip
    cfg = base_config(fake_project)
    with pytest.raises(QFError, match="qf train fetch-base"):
        verify_base_weights(cfg, fake_project)
    directory = download_base_weights(cfg, fake_project)
    (directory / "model-00001-of-00001.safetensors").write_bytes(b"short")
    with pytest.raises(QFError, match="missing or changed"):
        verify_base_weights(cfg, fake_project)


def test_files_not_on_the_hub_are_refused(fake_project: Path, fake_hub: dict[str, bytes]) -> None:
    with pytest.raises(QFError, match="has no"):
        download_base_weights(base_config(fake_project, ["model-9.safetensors"]), fake_project)
    with pytest.raises(QFError, match="no weight_files"):
        download_base_weights(base_config(fake_project, []), fake_project)


def test_cli_fetch_base(fake_project: Path, fake_hub: dict[str, bytes],
                        capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    base_config(fake_project)
    assert main(["train", "fetch-base"]) == 1  # 24.5 GB only with the human decision
    assert "needs the human decision" in capsys.readouterr().err
    assert main(["train", "fetch-base", "--approval", "автор проекта, 2026-09-25: разрешено"]) == 0
    out: Any = capsys.readouterr().out
    assert "Downloading 2 files of fake/model@aaaaaaaaaaaa" in out and "sha256 checked" in out
    assert "0.0 GB" in out
