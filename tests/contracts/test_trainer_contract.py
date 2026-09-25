"""Contract test of every registered adapter trainer (plan C.8, step 12).

Each trainer must be testable offline on CPU: `OFFLINE_CONFIGS` holds parameters that need no
GPU, download or real model (`hf_trainer_qlora`: the tiny random Mistral). A trainer registered
without an entry here fails `test_every_trainer_has_an_offline_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import qf.cli.wiring  # noqa: F401  (every registry and lazy entry)
from qf.cli.wiring import build
from qf.common import ComponentConfig, read_artifact
from qf.contracts import AdapterTrainer, Estimate
from qf.training import TRAINERS, data_refs, load_records, load_tokenizer, token_stats

OFFLINE_CONFIGS: dict[str, dict[str, Any]] = {
    "hf_trainer_qlora": {"loader": "tiny_random_cpu"},
}


def trainer(name: str) -> AdapterTrainer:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    instance = build(TRAINERS, ComponentConfig(name=name, **OFFLINE_CONFIGS[name]))
    assert isinstance(instance, AdapterTrainer)
    return instance


def test_every_trainer_has_an_offline_config() -> None:
    assert sorted(TRAINERS.names()) == sorted(OFFLINE_CONFIGS)


@pytest.mark.parametrize("name", TRAINERS.names())
def test_trainer_estimates_without_gpu(name: str, fake_project: Path) -> None:
    pytest.importorskip("tokenizers")
    from tests.tiny_training import tiny_config, tiny_project

    project = tiny_project(fake_project)
    cfg = tiny_config(project, trainer=OFFLINE_CONFIGS[name])
    tok = load_tokenizer(fake_project / "artifacts/base_model/fake-tiny")
    records = load_records(fake_project / cfg.data.train_path, cfg.data.expected_sha256.train)
    stats, skipped = token_stats(records, tok, cfg.training.max_sequence_length)
    est = trainer(name).estimate(cfg, stats, None)
    assert isinstance(est, Estimate) and skipped == []
    assert est.processed_tokens == stats.tokens_total and est.gpu_hours is None


@pytest.mark.slow
@pytest.mark.parametrize("name", TRAINERS.names())
def test_trainer_returns_a_lora_adapter(name: str, fake_project: Path) -> None:
    pytest.importorskip("tokenizers")
    from tests.tiny_training import tiny_config, tiny_project

    project = tiny_project(fake_project)
    cfg = tiny_config(project, max_steps=2, trainer=OFFLINE_CONFIGS[name])
    train_ref, val_ref = data_refs(cfg, fake_project)
    ref = trainer(name).train(cfg, train_ref, val_ref, fake_project / "runs/contract", None)
    assert ref.kind == "lora_adapter" and ref.schema_version == "peft_lora_v1"
    assert ref.parents == (train_ref.sha256, val_ref.sha256)
    assert read_artifact(fake_project / ref.path, "lora_adapter", {"peft_lora_v1"},
                         root=fake_project) == ref  # fmt: skip
