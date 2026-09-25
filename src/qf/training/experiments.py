"""Registered experiments (D-118): `configs/experiments/EXXX.yaml`, committed before the run.

An experiment fixes its training config by hash (the seed excluded); a run under the ID must
use exactly that config and one of the registered seeds. No command-line override can change
it: new parameters mean a new experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from qf.common import QFError, load_yaml_config, sha256_json
from qf.contracts import ExperimentSpec, TrainConfig
from qf.training.config import load_train_config

__all__ = [
    "EXPERIMENTS_DIR",
    "experiment_config_hash",
    "load_experiment",
    "registered_config",
    "replicates_done",
]

EXPERIMENTS_DIR: Final = Path("configs/experiments")


def experiment_config_hash(cfg: TrainConfig) -> str:
    """sha256 of the training config without its seed: the identity of an experiment."""
    data = cfg.model_dump(mode="json")
    data.pop("seed")
    return sha256_json(data)


def load_experiment(root: Path, experiment_id: str) -> ExperimentSpec:
    path = root / EXPERIMENTS_DIR / f"{experiment_id}.yaml"
    if not path.exists():
        raise QFError(f"experiment {experiment_id} is not registered: no {path.relative_to(root)} "
                      "(register it in Git before the run, D-118)")  # fmt: skip
    spec = load_yaml_config(path, ExperimentSpec)
    if spec.id != experiment_id:
        raise QFError(f"{path.name} registers {spec.id}, not {experiment_id}")
    return spec


def registered_config(root: Path, spec: ExperimentSpec, seed: int | None) -> TrainConfig:
    """The training config of the experiment with the chosen seed (default: the config's own),
    after checking that the config file still has the registered hash."""
    cfg = load_train_config(root / spec.config)
    actual = experiment_config_hash(cfg)
    if actual != spec.config_sha256:
        raise QFError(f"{spec.config} changed since {spec.id} was registered (hash {actual[:12]}…, "
                      f"registered {spec.config_sha256[:12]}…): new parameters need a new "
                      "experiment")  # fmt: skip
    chosen = cfg.seed if seed is None else seed
    if chosen not in spec.seeds:
        raise QFError(f"seed {chosen} is not registered for {spec.id} (seeds {spec.seeds})")
    return cfg if chosen == cfg.seed else cfg.model_copy(update={"seed": chosen})


def replicates_done(runs: Path, experiment_id: str) -> dict[int, str]:
    """{seed: run_id} of the completed runs of the experiment in this `runs/` directory (only
    this machine's runs are visible)."""
    done: dict[int, str] = {}
    for manifest in sorted(runs.glob("*/run_manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("experiment_id") == experiment_id and data.get("status") == "completed":
            done[int(data["seed"])] = data["run_id"]
    return done
