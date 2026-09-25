"""Resource monitoring of training runs (D-118)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from qf.training.trainers.hf_qlora.monitor import ResourceMonitor, parse_nvidia_smi  # noqa: E402


def test_parse_nvidia_smi() -> None:
    rows = parse_nvidia_smi("0, 97, 30210, 81920, 251.40, 61\n1, [N/A], 10, 81920, [N/A], 40\n")
    assert rows[0] == {
        "index": 0.0, "util_percent": 97.0, "memory_used_mib": 30210.0,
        "memory_total_mib": 81920.0, "power_w": 251.4, "temperature_c": 61.0,
    }  # fmt: skip
    assert rows[1]["util_percent"] is None and rows[1]["power_w"] is None
    assert parse_nvidia_smi("garbage\n") == []


def test_summary_integrates_energy(tmp_path: Path) -> None:
    powers = iter([200.0, 300.0, 300.0])

    def gpu() -> list[dict[str, float | None]]:
        row = {"index": 0.0, "util_percent": 90.0, "memory_used_mib": 1000.0,
               "memory_total_mib": 2000.0, "temperature_c": 50.0}  # fmt: skip
        return [{**row, "power_w": next(powers)}]

    monitor = ResourceMonitor(tmp_path / "resources.jsonl", 1.0, gpu_sampler=gpu)
    for elapsed in (0.0, 1800.0, 3600.0):  # two half-hours
        line = monitor.sample()
        line["elapsed_s"] = elapsed
    summary = monitor.summary()
    # (200+300)/2 W × 0.5 h + 300 W × 0.5 h = 275 Wh
    assert summary["gpu_energy_kwh"] == pytest.approx(0.275)
    assert summary["gpu_power_peak_w"] == 300.0 and summary["gpu_util_mean_percent"] == 90.0
    assert len((tmp_path / "resources.jsonl").read_text().splitlines()) == 3


def test_thread_samples_until_stopped(tmp_path: Path) -> None:
    monitor = ResourceMonitor(tmp_path / "r.jsonl", 0.01, gpu_sampler=lambda: None)
    monitor.step = 3
    monitor.start()
    monitor.stop()
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(lines) >= 2 and lines[-1]["step"] == 3 and lines[-1]["rss_bytes"] > 0
    assert monitor.summary()["gpu_energy_kwh"] is None  # no GPU: no energy
