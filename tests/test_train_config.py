"""The training config and its preflight checks (plan step 12)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qf.common import QFError, load_yaml_config, write_artifact
from qf.contracts import TrainConfig
from qf.training import (
    BaseModelConfig,
    base_model_for,
    data_refs,
    load_records,
    load_train_config,
    system_prompt_hash,
    with_overrides,
)
from tests.conftest import REPO_ROOT
from tests.factories import make_record

REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
VALID: dict[str, Any] = {
    "run_name": "test",
    "seed": 42,
    "model": {"id": "mistralai/Mistral-Nemo-Instruct-2407", "revision": REVISION},
    "quantization": {"load_in_4bit": True, "compute_dtype": "bf16"},
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                "down_proj"]},
    "data": {"train_path": "data/train.jsonl", "val_path": "data/val.jsonl",
             "expected_sha256": {"train": "a" * 64, "val": "b" * 64}},
    "training": {"max_sequence_length": 2048, "micro_batch_size": 1,
                 "gradient_accumulation_steps": 16, "learning_rate": 1e-4, "epochs": 1,
                 "max_steps": 94, "max_wall_time_minutes": 120, "warmup_ratio": 0.03,
                 "max_grad_norm": 1.0, "checkpoint_every_optimizer_steps": 10,
                 "evaluate_every_optimizer_steps": 10, "keep_checkpoints": 10},
    "trainer": {"name": "hf_trainer_qlora"},
}  # fmt: skip


def valid_config() -> dict[str, Any]:
    return copy.deepcopy(VALID)


def invalid(section: str, field: str, value: Any = None, *, drop: bool = False) -> str:
    data = valid_config()
    if drop:
        del data[section][field]
    else:
        data.setdefault(section, {})[field] = value
    with pytest.raises(ValidationError) as caught:
        TrainConfig.model_validate(data)
    return str(caught.value)


def test_valid_config() -> None:
    cfg = TrainConfig.model_validate(valid_config())
    assert cfg.training.max_steps == 94 and cfg.trainer.name == "hf_trainer_qlora"
    assert cfg.model.base_model_config == Path("configs/train/base_model.yaml")


def test_placeholder_revision_rejected() -> None:
    assert "placeholder" in invalid("model", "revision", "REQUIRED_PINNED_COMMIT")
    assert "40-hex" in invalid("model", "revision", "main")


def test_max_steps_required() -> None:
    assert "max_steps" in invalid("training", "max_steps", drop=True)
    assert "max_wall_time_minutes" in invalid("training", "max_wall_time_minutes", drop=True)
    invalid("training", "max_steps", 0)


def test_bfloat16_if_supported_string_rejected() -> None:
    assert "decide explicitly" in invalid("quantization", "compute_dtype", "bfloat16_if_supported")
    invalid("quantization", "compute_dtype", "fp32")


def test_unknown_keys_and_training_rules_rejected() -> None:
    invalid("training", "packing", True)
    invalid("training", "loss_scope", "all_tokens")
    invalid("logging", "external_reporting", True)
    data = valid_config()
    data["training"]["surprise"] = 1
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(data)


def test_overrides_are_validated() -> None:
    cfg = TrainConfig.model_validate(valid_config())
    smoke = with_overrides(cfg, max_steps=15, checkpoint_every_optimizer_steps=5,
                           evaluate_every_optimizer_steps=None)  # fmt: skip
    assert smoke.training.max_steps == 15 and smoke.training.checkpoint_every_optimizer_steps == 5
    assert smoke.training.evaluate_every_optimizer_steps == 10
    assert with_overrides(cfg) is cfg
    with pytest.raises(QFError, match="invalid override"):
        with_overrides(cfg, max_steps=-1)


def _dataset(root: Path) -> tuple[str, str]:
    hashes = []
    for split in ("train", "val"):
        path = root / "data" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = make_record(id=f"qf-{split}-LOAD00001-0", split=split)
        path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        hashes.append(write_artifact(path, "sft_dataset", "sft_record_v1", "r", root=root).sha256)
    return hashes[0], hashes[1]


def test_data_hash_mismatch_refuses(fake_project: Path) -> None:
    train_sha, val_sha = _dataset(fake_project)
    data = valid_config()
    data["data"]["expected_sha256"] = {"train": train_sha, "val": val_sha}
    train_ref, val_ref = data_refs(TrainConfig.model_validate(data), fake_project)
    assert (train_ref.sha256, val_ref.sha256) == (train_sha, val_sha)
    data["data"]["expected_sha256"]["val"] = "0" * 64
    with pytest.raises(QFError, match="data hash mismatch for val"):
        data_refs(TrainConfig.model_validate(data), fake_project)
    with pytest.raises(QFError, match="differs from the expected"):
        load_records(fake_project / "data/train.jsonl", "0" * 64)
    (fake_project / "data/train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        data_refs(TrainConfig.model_validate(data), fake_project)


def test_system_prompt_hash_needs_one_prompt() -> None:
    record = make_record()
    assert system_prompt_hash([record, record]) == system_prompt_hash([record])
    other = record.model_copy(
        update={
            "messages": [
                record.messages[0].model_copy(update={"content": "другой промпт"}),
                *record.messages[1:],
            ]
        }
    )
    with pytest.raises(QFError, match="found 2"):
        system_prompt_hash([record, other])


def test_base_model_config_must_match(fake_project: Path) -> None:
    config = fake_project / "configs/train/base_model.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(f"id: mistralai/Mistral-Nemo-Instruct-2407\nrevision: {'c' * 40}\n"
                      "license: apache-2.0\ntokenizer_files: [tokenizer.json]\n"
                      "local_dir: artifacts/base\n", encoding="utf-8")  # fmt: skip
    with pytest.raises(QFError, match="pins"):
        base_model_for(TrainConfig.model_validate(valid_config()), fake_project)


def test_repository_configs_are_valid_and_pinned() -> None:
    pilot = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml")
    tiny = load_train_config(REPO_ROOT / "configs/train/tiny_cpu_test.yaml")
    base = load_yaml_config(REPO_ROOT / "configs/train/base_model.yaml", BaseModelConfig)
    for cfg in (pilot, tiny):
        assert (cfg.model.id, cfg.model.revision) == (base.id, base.revision)
    assert pilot.trainer.model_extra == {"loader": "cuda_4bit"}
    assert tiny.trainer.model_extra == {"loader": "tiny_random_cpu"}
    assert pilot.training.max_steps == 94 and pilot.training.epochs == 1
    assert pilot.quantization.load_in_4bit and pilot.quantization.compute_dtype == "bf16"
    assert pilot.data.train_path.as_posix() == "data/processed/generated_v2/train.jsonl"
    assert "consolidated.safetensors" not in base.weight_files
    assert len([f for f in base.weight_files if f.endswith(".safetensors")]) == 5


def test_pilot_config_hashes_match_the_dataset() -> None:
    pilot = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml")
    if not (REPO_ROOT / pilot.data.train_path).exists():
        pytest.skip("generated_v2 is not built")
    train_ref, val_ref = data_refs(pilot, REPO_ROOT)
    assert train_ref.sha256 == pilot.data.expected_sha256.train
