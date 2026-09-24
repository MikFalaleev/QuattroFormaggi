"""Read-only environment report (`qf doctor`).

No downloads, no GPU computation, no secret values: only facts about this machine, installed
packages and local services. Hardware serial numbers are never collected.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from qf.common._optional import import_optional, is_installed
from qf.common.errors import QFError
from qf.common.manifest import collect_git_info, collect_package_versions
from qf.common.paths import RUNS, project_root

__all__ = [
    "DEFAULT_REPORT_PATH",
    "LMSTUDIO_MODELS_URL",
    "collect_environment",
    "collect_hardware",
    "format_environment_report",
]

DEFAULT_REPORT_PATH = RUNS / "doctor" / "hardware.json"
LMSTUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
LMSTUDIO_TIMEOUT_S = 2.0
TRAINING_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
HF_TOKEN_ENV = "HF_TOKEN"
LLAMA_CPP_DIR = Path("third_party/llama.cpp")
PROJECT_DISTRIBUTION = "quattro-formaggi"


def _platform_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python_version": platform.python_version(),
    }


def _chip() -> str | None:
    """The processor name, e.g. "Apple M4 Max" (macOS); None where it cannot be read."""
    if platform.system() != "Darwin":
        return platform.processor() or None
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
                                capture_output=True, text=True, timeout=5, check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_hardware() -> dict[str, Any]:
    """What a run manifest records about the machine (eval runs, D-108): platform, processor,
    cores and memory. Read-only and fast; nothing about disks or packages."""
    memory = psutil.virtual_memory()
    return {
        **_platform_info(),
        "chip": _chip(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": memory.total,
    }


def _resource_info(root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(root)
    return {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_path": str(root),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _package_info() -> dict[str, dict[str, Any]]:
    versions = collect_package_versions(TRAINING_PACKAGES)
    return {
        name: {"installed": is_installed(name), "version": versions[name]}
        for name in TRAINING_PACKAGES
    }


def _torch_info() -> dict[str, Any]:
    # Imported by name at runtime: qf.common must not import torch statically (plan C.3).
    torch = import_optional("torch")
    if torch is None:
        return {"installed": False}
    try:
        cuda = bool(torch.cuda.is_available())
        devices = [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count() if cuda else 0)
        ]
        mps = bool(torch.backends.mps.is_available())
    except Exception as exc:  # report, never crash: doctor is diagnostic
        return {"installed": True, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "installed": True,
        "cuda_available": cuda,
        "cuda_devices": devices,
        "mps_available": mps,
    }


def _probe_lmstudio(transport: httpx.BaseTransport | None) -> dict[str, Any]:
    result: dict[str, Any] = {"url": LMSTUDIO_MODELS_URL, "available": False}
    try:
        with httpx.Client(transport=transport, timeout=LMSTUDIO_TIMEOUT_S, trust_env=False) as c:
            response = c.get(LMSTUDIO_MODELS_URL)
    except httpx.HTTPError as exc:
        result["error"] = type(exc).__name__
        return result
    if response.status_code != httpx.codes.OK:
        result["error"] = f"HTTP {response.status_code}"
        return result
    result["available"] = True
    try:
        result["models"] = [item["id"] for item in response.json()["data"]]
    except (ValueError, KeyError, TypeError):
        result["models"] = None
    return result


def _llama_cpp_info(root: Path) -> dict[str, Any]:
    path = root / LLAMA_CPP_DIR
    if not (path / ".git").exists():
        return {"present": path.is_dir(), "path": str(LLAMA_CPP_DIR), "commit": None}
    commit, dirty = collect_git_info(path)
    return {"present": True, "path": str(LLAMA_CPP_DIR), "commit": commit, "dirty": dirty}


def _default_root() -> Path:
    try:
        return project_root()
    except QFError:
        return Path.cwd()


def collect_environment(
    *, root: Path | None = None, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Collect the read-only environment report. `transport` lets tests stub LM Studio."""
    root = root if root is not None else _default_root()
    lms_path = shutil.which("lms")
    return {
        "report_version": "doctor_v1",
        "collected_at": datetime.now(UTC).isoformat(),
        "qf_version": collect_package_versions([PROJECT_DISTRIBUTION])[PROJECT_DISTRIBUTION],
        "platform": _platform_info(),
        "resources": _resource_info(root),
        "packages": _package_info(),
        "torch": _torch_info(),
        "lmstudio": _probe_lmstudio(transport),
        "lms_cli": {"found": lms_path is not None, "path": lms_path},
        "llama_cpp": _llama_cpp_info(root),
        "secrets": {"hf_token_set": bool(os.environ.get(HF_TOKEN_ENV))},
    }


def _gib(value: int) -> str:
    return f"{value / 2**30:.1f} GiB"


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def format_environment_report(env: dict[str, Any]) -> str:
    """Human-readable summary of `collect_environment()` output."""
    plat, res, torch_info = env["platform"], env["resources"], env["torch"]
    lines = [
        f"qf {env['qf_version']} — environment report ({env['collected_at']})",
        f"Platform: {plat['platform']} ({plat['machine']}), Python {plat['python_version']}",
        f"CPU: {res['cpu_count_physical']} physical / {res['cpu_count_logical']} logical cores",
        f"RAM: {_gib(res['ram_total_bytes'])} total, {_gib(res['ram_available_bytes'])} available",
        f"Disk ({res['disk_path']}): {_gib(res['disk_free_bytes'])} free "
        f"of {_gib(res['disk_total_bytes'])}",
        "Training packages:",
    ]
    for name, info in env["packages"].items():
        lines.append(f"  {name}: {info['version'] if info['installed'] else 'not installed'}")
    if torch_info.get("installed"):
        if "error" in torch_info:
            lines.append(f"torch probe failed: {torch_info['error']}")
        else:
            lines.append(
                f"CUDA available: {_yes_no(torch_info['cuda_available'])}; "
                f"MPS available: {_yes_no(torch_info['mps_available'])}"
            )
            for device in torch_info["cuda_devices"]:
                lines.append(f"  GPU: {device['name']} ({_gib(device['total_memory_bytes'])})")
    lms = env["lmstudio"]
    lms_detail = f"models: {lms.get('models')}" if lms["available"] else lms.get("error", "")
    lines += [
        f"LM Studio server ({lms['url']}): {_yes_no(lms['available'])} {lms_detail}".rstrip(),
        f"lms CLI in PATH: {_yes_no(env['lms_cli']['found'])}",
        f"llama.cpp ({env['llama_cpp']['path']}): present={_yes_no(env['llama_cpp']['present'])}, "
        f"commit={env['llama_cpp']['commit']}",
        f"HF_TOKEN set: {_yes_no(env['secrets']['hf_token_set'])}",
    ]
    return "\n".join(lines)
