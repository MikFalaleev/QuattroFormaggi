"""Training budget arithmetic and the refusals of `qf train run` (plan step 12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qf.common import QFError
from qf.contracts import TokenStats, TrainConfig
from qf.training import (
    MistralDims,
    base_weight_bytes,
    check_budget,
    check_max_steps,
    estimate,
    steps_for_epochs,
)
from tests.conftest import REPO_ROOT
from tests.test_lora_param_count import NEMO_CONFIG
from tests.test_train_config import valid_config

STATS = TokenStats(records=1500, skipped_too_long=0, tokens_total=1_914_735,
                   tokens_trainable=218_000, tokens_max=1438, tokens_mean=1276.5)  # fmt: skip
NEMO = MistralDims.from_config(NEMO_CONFIG)


def config(**changes: object) -> TrainConfig:
    data = valid_config()
    for key, value in changes.items():
        section, field = key.split("__")
        data.setdefault(section, {})[field] = value
    return TrainConfig.model_validate(data)


def test_steps_math() -> None:
    assert steps_for_epochs(1500, 1, 16, 1) == 94
    assert steps_for_epochs(1500, 1, 16, 2) == 188
    assert steps_for_epochs(10, 1, 4, 1) == 3  # the last step is shorter
    assert steps_for_epochs(16, 16, 1, 30) == 30
    with pytest.raises(ValueError):
        steps_for_epochs(0, 1, 16, 1)


def test_max_steps_above_one_epoch_rejected() -> None:
    check_max_steps(config(training__max_steps=94), 1500)
    check_max_steps(config(training__max_steps=15), 1500)  # a partial pass is allowed
    with pytest.raises(QFError, match="max_steps 95 exceeds 94"):
        check_max_steps(config(training__max_steps=95), 1500)


def test_tokens_counted_all_not_only_trainable() -> None:
    est = estimate(config(training__max_steps=94), STATS, 1000.0, NEMO)
    assert est.processed_tokens == 1_914_735 and est.trainable_tokens == 218_000
    assert est.gpu_hours == pytest.approx(1_914_735 / 1000 / 3600)
    partial = estimate(config(training__max_steps=15), STATS, None, NEMO)
    assert partial.optimizer_steps == 15
    assert partial.processed_tokens == round(1_914_735 * 15 / 94)
    assert any("partial" in note for note in partial.notes)


def test_cost_none_without_tps() -> None:
    est = estimate(config(budget__gpu_hourly_rate=231.17), STATS, None, NEMO)
    assert est.tokens_per_s is None and est.gpu_hours is None and est.cost is None
    assert est.cost_ceiling == pytest.approx(120 / 60 * 231.17)  # the wall-time ceiling
    assert est.notes and "not measured" in est.notes[0]


def test_cost_with_tps_includes_overhead() -> None:
    est = estimate(config(budget__gpu_hourly_rate=200.0), STATS, 2000.0, NEMO)
    assert est.cost == pytest.approx(1_914_735 / 2000 / 3600 * 200.0 * 1.3)


def test_run_refuses_over_budget() -> None:
    fits = config(budget__gpu_hourly_rate=231.17, budget__max_cost=500.0)
    check_budget(estimate(fits, STATS, None, NEMO), fits)
    ceiling = config(budget__gpu_hourly_rate=231.17, budget__max_cost=400.0)
    with pytest.raises(QFError, match="over budget: max_wall_time_minutes 120"):
        check_budget(estimate(ceiling, STATS, None, NEMO), ceiling)
    slow = config(budget__gpu_hourly_rate=231.17, budget__max_cost=460.0,
                  training__max_wall_time_minutes=100)  # fmt: skip
    with pytest.raises(QFError, match="estimated cost"):
        check_budget(estimate(slow, STATS, 100.0, NEMO), slow)  # 5.3 h at 100 tokens/s
    no_rate = config(budget__max_cost=100.0)
    with pytest.raises(QFError, match="gpu_hourly_rate is not"):
        check_budget(estimate(no_rate, STATS, None, NEMO), no_rate)
    check_budget(estimate(config(), STATS, None, NEMO), config())  # no limit, no refusal


def test_memory_estimate() -> None:
    assert NEMO is not None
    est = estimate(config(), STATS, None, NEMO)
    memory = est.memory
    assert memory.base_weights_plan_bytes == round(12.2e9 * 0.5 * 1.1)
    # 10.9e9 nf4 parameters + 1.34e9 fp32 embeddings and lm_head: more than the plan's formula
    assert memory.base_weights_bytes == base_weight_bytes(NEMO)
    assert 10.5e9 < memory.base_weights_bytes < 11.5e9  # type: ignore[operator]
    assert memory.lora_bytes == 57_016_320 * 16
    assert memory.logits_bytes == 1438 * 131072 * 4 * 3
    assert memory.activations_bytes is None


def test_unknown_model_config_gives_no_memory_or_lora() -> None:
    assert MistralDims.from_config({}) is None
    est = estimate(config(), STATS, None, None)
    assert est.lora_trainable_params is None and est.memory.base_weights_bytes is None


@pytest.mark.needs_tokenizer
def test_estimate_reproduces_the_mask_audit() -> None:
    """The estimate counts tokens exactly like the loss-mask audit of step 11 (E200)."""
    pytest.importorskip("transformers")
    from qf.cli.commands.train import _stats
    from qf.training import load_train_config

    cfg = load_train_config(REPO_ROOT / "configs/train/qlora_nemo_v0.1.yaml")
    if not (REPO_ROOT / cfg.data.train_path).exists():
        pytest.skip("generated_v2 is not built")
    stats = _stats(cfg, REPO_ROOT)
    assert (stats.records, stats.tokens_total, stats.tokens_trainable) == (1500, 1_914_735,
                                                                           218_000)  # fmt: skip
    config_json = cfg.model.base_model_config
    base = json.loads((REPO_ROOT / "artifacts/base_model/mistralai--Mistral-Nemo-Instruct-2407"
                       / cfg.model.revision / "config.json").read_text("utf-8"))  # fmt: skip
    assert MistralDims.from_config(base) == NEMO and Path(config_json).name == "base_model.yaml"
