"""Integrity of a trained adapter and its run (D-118): after training, and on the Mac after the
run is copied off the GPU machine, before the machine is deleted (`qf train verify`).

The safetensors file is read directly (header + raw little-endian data) with numpy, so the check
needs no torch and works on any machine.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from qf.common import QFError, read_artifact, read_manifest
from qf.contracts import supported_versions

__all__ = [
    "INTEGRITY_FILE",
    "AdapterCheck",
    "read_safetensors",
    "verify_adapter",
    "verify_run",
]

INTEGRITY_FILE: Final = "integrity.json"
_DTYPES: Final[dict[str, Any]] = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}


@dataclass
class AdapterCheck:
    checks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    tensors: int = 0
    parameters: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "problems": self.problems,
                "tensors": self.tensors, "parameters": self.parameters}  # fmt: skip


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Tensors of a safetensors file as float32 numpy arrays (F32, F16, BF16)."""
    data = path.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    body = data[8 + length :]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES.get(info["dtype"])
        if dtype is None:
            raise QFError(f"{path.name}: tensor {name} has unsupported dtype {info['dtype']}")
        start, end = info["data_offsets"]
        raw = np.frombuffer(body[start:end], dtype=dtype)
        if info["dtype"] == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        tensors[name] = raw.astype(np.float32).reshape(info["shape"])
    return tensors


def verify_adapter(adapter_dir: Path, root: Path, *, expected_parameters: int | None,
                   base_model: str, revision: str | None) -> AdapterCheck:  # fmt: skip
    """Hash against the artifact manifest, adapter_config naming the base, only LoRA tensors,
    the expected number of parameters, all values finite."""
    check = AdapterCheck()
    try:
        read_artifact(adapter_dir, "lora_adapter", supported_versions("lora_adapter"), root=root)
        check.checks.append("artifact sha256 matches its manifest")
    except QFError as exc:
        check.problems.append(str(exc))
        return check
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    named = (config.get("base_model_name_or_path"), config.get("revision"))
    if named != (base_model, revision):
        check.problems.append(f"adapter_config names {named[0]}@{named[1]}, expected "
                              f"{base_model}@{revision}")  # fmt: skip
    else:
        check.checks.append("adapter_config names the base model and revision")
    tensors = read_safetensors(adapter_dir / "adapter_model.safetensors")
    check.tensors = len(tensors)
    check.parameters = sum(t.size for t in tensors.values())
    foreign = [n for n in tensors if "lora_A" not in n and "lora_B" not in n]
    if foreign:
        check.problems.append(f"tensors outside LoRA: {foreign[:5]}")
    else:
        check.checks.append(f"{check.tensors} tensors, all LoRA A/B")
    if expected_parameters is not None and check.parameters != expected_parameters:
        check.problems.append(f"{check.parameters} parameters, expected {expected_parameters}")
    else:
        check.checks.append(f"{check.parameters} parameters")
    broken = [n for n, t in tensors.items() if not np.isfinite(t).all()]
    if broken:
        check.problems.append(f"NaN or inf in {broken[:5]}")
    else:
        check.checks.append("all values finite")
    return check


def verify_run(run_dir: Path, root: Path) -> AdapterCheck:
    """A completed training run as copied: manifest, logs and the adapter."""
    manifest = read_manifest(run_dir)
    if manifest.status != "completed":
        check = AdapterCheck()
        check.problems.append(f"run status is {manifest.status}, not completed")
        return check
    adapter_manifest = json.loads((run_dir / "adapter" / "qf_adapter_manifest.json")
                                  .read_text(encoding="utf-8"))  # fmt: skip
    trained_on = adapter_manifest["trained_on"]
    check = verify_adapter(run_dir / "adapter", root,
                           expected_parameters=manifest.metrics.get("trainable_params"),
                           base_model=trained_on,
                           revision=adapter_manifest["revision"]
                           if trained_on == adapter_manifest["base_model"] else None)  # fmt: skip
    for name in ("train_log.jsonl", "train_config.json", "events.jsonl"):
        if (run_dir / name).exists():
            check.checks.append(f"{name} present")
        else:
            check.problems.append(f"{name} is missing")
    return check
