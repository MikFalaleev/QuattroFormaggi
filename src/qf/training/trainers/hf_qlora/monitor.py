"""Resource monitoring of a training run (D-118): a background thread samples the GPU (via
`nvidia-smi`), torch's CUDA memory and the process every N seconds into `resources.jsonl`; the
run manifest gets the peaks and the GPU energy in kWh."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import psutil
import torch

__all__ = ["NVIDIA_QUERY", "RESOURCES_FILE", "ResourceMonitor", "nvidia_smi", "parse_nvidia_smi"]

RESOURCES_FILE: Final = "resources.jsonl"
NVIDIA_QUERY: Final = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
_FIELDS: Final = ("index", "util_percent", "memory_used_mib", "memory_total_mib", "power_w",
                  "temperature_c")  # fmt: skip


def parse_nvidia_smi(text: str) -> list[dict[str, float | None]]:
    """Rows of `nvidia-smi --query-gpu=<NVIDIA_QUERY> --format=csv,noheader,nounits`."""
    rows = []
    for line in text.strip().splitlines():
        values: list[float | None] = []
        for cell in line.split(","):
            try:
                values.append(float(cell.strip()))
            except ValueError:  # "[N/A]", "[Not Supported]"
                values.append(None)
        if len(values) == len(_FIELDS):
            rows.append(dict(zip(_FIELDS, values, strict=True)))
    return rows


def nvidia_smi() -> list[dict[str, float | None]] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(result.stdout) if result.returncode == 0 else None


class ResourceMonitor:
    """Samples every `interval_s` seconds until `stop()`; `step` is set by the trainer."""

    def __init__(self, path: Path, interval_s: float,
                 gpu_sampler: Callable[[], list[dict[str, float | None]] | None] = nvidia_smi,
                 ) -> None:  # fmt: skip
        self.path = path
        self.interval_s = interval_s
        self.gpu_sampler = gpu_sampler
        self.step = 0
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qf-resources", daemon=True)
        self._process = psutil.Process()
        self._started = time.monotonic()

    def sample(self) -> dict[str, Any]:
        line: dict[str, Any] = {"time": time.time(),
                                "elapsed_s": round(time.monotonic() - self._started, 3),
                                "step": self.step, "gpus": self.gpu_sampler(),
                                "rss_bytes": self._process.memory_info().rss,
                                "cpu_percent": self._process.cpu_percent(),
                                "ram_used_bytes": psutil.virtual_memory().used}  # fmt: skip
        if torch.cuda.is_available():
            line.update(torch_allocated_bytes=torch.cuda.memory_allocated(),
                        torch_reserved_bytes=torch.cuda.memory_reserved(),
                        torch_max_allocated_bytes=torch.cuda.max_memory_allocated())  # fmt: skip
        self.samples.append(line)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(line) + "\n")
        return line

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s + 30)
        self.sample()

    def summary(self) -> dict[str, Any]:
        """Peaks, mean GPU utilisation and GPU energy (trapezoids of power over time)."""
        gpus = [(s["elapsed_s"], s["gpus"]) for s in self.samples if s.get("gpus")]
        energy_wh = 0.0
        for (t0, g0), (t1, g1) in zip(gpus, gpus[1:], strict=False):
            for a, b in zip(g0, g1, strict=False):
                if a["power_w"] is not None and b["power_w"] is not None:
                    energy_wh += (a["power_w"] + b["power_w"]) / 2 * (t1 - t0) / 3600

        def values(key: str) -> list[float]:
            return [g[key] for _, row in gpus for g in row if g[key] is not None]

        util = values("util_percent")
        return {
            "resource_samples": len(self.samples),
            "rss_peak_bytes": max((s["rss_bytes"] for s in self.samples), default=None),
            "gpu_memory_used_peak_mib": max(values("memory_used_mib"), default=None),
            "gpu_util_mean_percent": round(sum(util) / len(util), 1) if util else None,
            "gpu_power_peak_w": max(values("power_w"), default=None),
            "gpu_temperature_peak_c": max(values("temperature_c"), default=None),
            "gpu_energy_kwh": round(energy_wh / 1000, 4) if gpus else None,
        }
