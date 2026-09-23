from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from qf.cli import main
from qf.common import collect_environment, format_environment_report

doctor_module = importlib.import_module("qf.common.doctor")
doctor_command = importlib.import_module("qf.cli.commands.doctor")


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


LMSTUDIO_DOWN = httpx.MockTransport(_refuse)


def _stub_cli_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.BaseTransport) -> None:
    real = doctor_command.collect_environment
    monkeypatch.setattr(doctor_command, "collect_environment", lambda: real(transport=transport))


def _run_doctor(tmp_path: Path) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "hardware.json"
    code = main(["doctor", "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_doctor_runs_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_module, "is_installed", lambda name: False)
    monkeypatch.setattr(doctor_module, "import_optional", lambda name: None)
    _stub_cli_transport(monkeypatch, LMSTUDIO_DOWN)
    code, report = _run_doctor(tmp_path)
    assert code == 0
    assert report["torch"] == {"installed": False}
    assert all(not info["installed"] for info in report["packages"].values())


def test_doctor_does_not_leak_hf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret123")
    _stub_cli_transport(monkeypatch, LMSTUDIO_DOWN)
    code, report = _run_doctor(tmp_path)
    captured = capsys.readouterr()
    assert code == 0
    assert report["secrets"] == {"hf_token_set": True}
    assert "secret123" not in captured.out + captured.err
    assert "secret123" not in (tmp_path / "hardware.json").read_text(encoding="utf-8")


def test_doctor_lmstudio_unreachable_is_not_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = collect_environment(root=tmp_path, transport=LMSTUDIO_DOWN)
    assert env["lmstudio"]["available"] is False
    assert env["lmstudio"]["error"] == "ConnectError"
    _stub_cli_transport(monkeypatch, LMSTUDIO_DOWN)
    code, report = _run_doctor(tmp_path)
    assert code == 0
    assert report["lmstudio"]["available"] is False


def test_doctor_lmstudio_available_lists_models(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "mistral-nemo"}]})
    )
    env = collect_environment(root=tmp_path, transport=transport)
    assert env["lmstudio"] == {
        "url": doctor_module.LMSTUDIO_MODELS_URL,
        "available": True,
        "models": ["mistral-nemo"],
    }
    assert "mistral-nemo" in format_environment_report(env)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(500), {"available": False, "error": "HTTP 500"}),
        (httpx.Response(200, text="not json"), {"available": True, "models": None}),
    ],
)
def test_doctor_lmstudio_odd_responses(
    tmp_path: Path, response: httpx.Response, expected: dict[str, Any]
) -> None:
    transport = httpx.MockTransport(lambda request: response)
    env = collect_environment(root=tmp_path, transport=transport)
    for key, value in expected.items():
        assert env["lmstudio"][key] == value


def _fake_torch(*, fail: bool = False) -> SimpleNamespace:
    def is_available() -> bool:
        if fail:
            raise RuntimeError("driver mismatch")
        return True

    cuda = SimpleNamespace(
        is_available=is_available,
        device_count=lambda: 1,
        get_device_name=lambda index: "FakeGPU 48GB",
        get_device_properties=lambda index: SimpleNamespace(total_memory=48 * 2**30),
    )
    return SimpleNamespace(
        cuda=cuda, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )


def test_doctor_reports_gpu_via_fake_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor_module, "import_optional", lambda name: _fake_torch())
    env = collect_environment(root=tmp_path, transport=LMSTUDIO_DOWN)
    assert env["torch"]["cuda_available"] is True
    assert env["torch"]["cuda_devices"] == [
        {"name": "FakeGPU 48GB", "total_memory_bytes": 48 * 2**30}
    ]
    assert "GPU: FakeGPU 48GB (48.0 GiB)" in format_environment_report(env)


def test_doctor_reports_torch_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor_module, "import_optional", lambda name: _fake_torch(fail=True))
    env = collect_environment(root=tmp_path, transport=LMSTUDIO_DOWN)
    assert env["torch"] == {"installed": True, "error": "RuntimeError: driver mismatch"}
    assert "torch probe failed" in format_environment_report(env)


def test_doctor_reads_llama_cpp_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    llama = tmp_path / "third_party" / "llama.cpp"
    llama.mkdir(parents=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(llama)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "pin"], check=True)
    env = collect_environment(root=tmp_path, transport=LMSTUDIO_DOWN)
    assert env["llama_cpp"]["present"] is True
    assert len(env["llama_cpp"]["commit"]) == 40


def test_doctor_report_has_no_hardware_identifiers(tmp_path: Path) -> None:
    text = json.dumps(collect_environment(root=tmp_path, transport=LMSTUDIO_DOWN)).lower()
    for forbidden in ("serial", "uuid", "udid"):
        assert forbidden not in text


def test_doctor_default_output_path(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cli_transport(monkeypatch, LMSTUDIO_DOWN)
    assert main(["doctor"]) == 0
    assert (fake_project / "runs" / "doctor" / "hardware.json").is_file()


def test_doctor_outside_project_uses_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_project() -> Path:
        raise doctor_module.QFError("no project")

    monkeypatch.setattr(doctor_module, "project_root", no_project)
    monkeypatch.chdir(tmp_path)
    env = collect_environment(transport=LMSTUDIO_DOWN)
    assert env["resources"]["disk_path"] == str(tmp_path)
