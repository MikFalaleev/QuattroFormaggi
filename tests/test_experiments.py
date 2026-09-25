"""Registered experiments (D-118): the committed files match their configs; runs are grouped."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qf.common import QFError
from qf.contracts import ExperimentSpec
from qf.training import (
    EXPERIMENTS_DIR,
    experiment_config_hash,
    load_experiment,
    load_train_config,
    registered_config,
    replicates_done,
)
from tests.conftest import REPO_ROOT


def test_repository_experiments_match_their_configs() -> None:
    files = sorted((REPO_ROOT / EXPERIMENTS_DIR).glob("E*.yaml"))
    assert [f.stem for f in files][:2] == ["E301", "E302"]
    for path in files:
        spec = load_experiment(REPO_ROOT, path.stem)
        cfg = registered_config(REPO_ROOT, spec, None)  # raises if the config changed
        assert experiment_config_hash(cfg) == spec.config_sha256
        assert cfg.seed in spec.seeds and spec.requires_approval


def test_pilot_and_smoke_differ_only_in_the_run_length() -> None:
    smoke = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1_smoke.yaml")
    pilot = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml")
    a, b = smoke.model_dump(mode="json"), pilot.model_dump(mode="json")
    changed = {f"{section}.{key}" for section in a if isinstance(a[section], dict)
               for key in a[section] if a[section][key] != b[section].get(key)}  # fmt: skip
    changed |= {key for key in a if not isinstance(a[key], dict) and a[key] != b[key]}
    assert changed == {"run_name", "training.max_steps", "training.max_wall_time_minutes",
                       "training.checkpoint_every_optimizer_steps",
                       "training.evaluate_every_optimizer_steps", "training.keep_checkpoints",
                       "budget.max_cost"}  # fmt: skip


def test_the_seed_is_not_part_of_the_identity() -> None:
    cfg = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml")
    other_seed = cfg.model_copy(update={"seed": 43})
    other_lr = cfg.model_copy(update={"training": cfg.training.model_copy(
        update={"learning_rate": 2e-4})})  # fmt: skip
    assert experiment_config_hash(other_seed) == experiment_config_hash(cfg)
    assert experiment_config_hash(other_lr) != experiment_config_hash(cfg)
    spec = load_experiment(REPO_ROOT, "E302")
    assert registered_config(REPO_ROOT, spec, 43).seed == 43
    with pytest.raises(QFError, match="seed 45 is not registered"):
        registered_config(REPO_ROOT, spec, 45)


def test_spec_validation(fake_project: Path) -> None:
    base = {"id": "E1", "title": "t", "registered": "2026-09-25", "plan_step": "14",
            "question": "q", "hypotheses": ["h"], "success": "s", "config": "c.yaml",
            "config_sha256": "a" * 64, "seeds": [1, 1]}  # fmt: skip
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(base)
    (fake_project / EXPERIMENTS_DIR).mkdir(parents=True)
    (fake_project / EXPERIMENTS_DIR / "E777.yaml").write_text(
        "id: E778\ntitle: t\nregistered: 2026-09-25\nplan_step: '1'\nquestion: q\n"
        "hypotheses: [h]\nsuccess: s\nconfig: c.yaml\nconfig_sha256: " + "a" * 64
        + "\nseeds: [1]\n", "utf-8")  # fmt: skip
    with pytest.raises(QFError, match="registers E778, not E777"):
        load_experiment(fake_project, "E777")


def test_replicates_done(tmp_path: Path) -> None:
    for run_id, experiment, seed, status in (("r1", "E302", 42, "completed"),
                                             ("r2", "E302", 43, "failed"),
                                             ("r3", "E301", 42, "completed")):  # fmt: skip
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "run_manifest.json").write_text(
            json.dumps(
                {"run_id": run_id, "experiment_id": experiment, "seed": seed, "status": status}
            )
        )
    assert replicates_done(tmp_path, "E302") == {42: "r1"}
