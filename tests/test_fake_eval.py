"""The builder of the step 9 acceptance run (`tests/fake_eval.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qf.cli.main import main
from qf.common import load_yaml_config
from qf.eval import METRICS, EvalConfig
from tests import fake_eval
from tests.test_harness import bench_records, install_bench


def test_acceptance_configs_list_every_metric() -> None:
    """bench_v1 keeps the metrics of step 9 (its run is pinned in tests/test_v1_frozen.py);
    bench_v2 adds the metrics of the special conditions."""
    assert set(fake_eval.METRICS) | set(fake_eval.METRICS_V2) == set(METRICS.names())
    assert len(fake_eval.METRICS) == 26 and "field_accuracy.temperature_c" in fake_eval.METRICS
    assert "field_accuracy.temperature_c" not in fake_eval.METRICS_V2
    assert set(fake_eval.SLICE_METRICS) <= set(fake_eval.METRICS)
    assert set(fake_eval.SLICE_METRICS_V2) <= set(fake_eval.METRICS_V2)


def test_mistake_pattern() -> None:
    gold = json.dumps({"card": {"pieces": None, "origin": {"city": "A", "region": "a"},
                                "destination": {"city": "B", "region": "b"}},
                       "missing_fields": ["pieces"], "conflicts": []})  # fmt: skip
    assert fake_eval.mistake(7, gold) == ("timeout", "__timeout__")
    assert fake_eval.mistake(5, gold)[0] == "invalid_json"
    kind, answer = fake_eval.mistake(0, gold)
    assert kind == "invented_pieces"
    assert json.loads(answer)["card"]["pieces"] == 1 and json.loads(answer)["missing_fields"] == []
    kind, answer = fake_eval.mistake(3, gold)
    assert kind == "od_swap" and json.loads(answer)["card"]["origin"]["city"] == "B"
    assert fake_eval.mistake(11, gold)[0] == "code_fence"
    assert fake_eval.mistake(9, gold) == ("gold", gold)


def test_build_and_run(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = bench_records()
    install_bench(fake_project, records)
    bench = Path("data/splits/bench_t.jsonl")
    assert fake_eval.main(["--bench", str(bench)]) == 0
    assert "Answers:" in capsys.readouterr().out
    out = fake_project / fake_eval.DEFAULT_OUT
    for name in ("fake_bench_t.yaml", "fake_bench_t_gold.yaml"):
        cfg = load_yaml_config(out / name, EvalConfig)
        assert cfg.bench.path == bench
        assert main(["eval", "run", "--config", str(out / name)]) == 0
    assert "4 cases, 0 failed" in capsys.readouterr().out
