from __future__ import annotations

import dataclasses
import subprocess
import sys
import tomllib

import pytest

import qf
from qf.cli import main
from qf.cli.main import COMMANDS
from qf.common import QFError
from tests.conftest import REPO_ROOT

PLANNED_COMMANDS = {
    "doctor",
    "data fetch",
    "data profile",
    "data facts",
    "data facts-v2",
    "data build",
    "data split",
    "validate-data",
    "data report",
    "bench freeze",
    "bench export-review",
    "bench import-review",
    "bench verify",
    "eval run",
    "eval-baseline",
    "eval compare",
    "ui",  # local web interface for manual testing, by the user's decision (D-107)
    "tokens fetch",
    "tokens audit",
    "train estimate",
    "train run",
    "train fetch-base",
    "train verify",
    "export merge",
    "export gguf",
    "export verify",
    "extract",
    "release check",
    "lab",
}
UNIMPLEMENTED = sorted(name for name, spec in COMMANDS.items() if not spec.implemented)


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"qf {qf.__version__}"


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert qf.__version__ == pyproject["project"]["version"]


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    help_text = " ".join(capsys.readouterr().out.split())  # argparse wraps long lines
    assert "[step 18, not implemented yet]" in help_text
    assert "validate datasets: schema, leakage, duplicates [step 7]" in help_text


def test_every_planned_command_registered() -> None:
    assert set(COMMANDS) == PLANNED_COMMANDS
    assert COMMANDS["doctor"].implemented


@pytest.mark.parametrize("name", UNIMPLEMENTED)
def test_unimplemented_command_exits_2(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(name.split()) == 2
    err = capsys.readouterr().err
    assert "not implemented" in err
    assert f"Stage {COMMANDS[name].stage} " in err


@pytest.mark.parametrize("name", ["export merge", "export gguf", "lab"])
def test_unimplemented_command_with_arguments_still_exits_2(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([*name.split(), "--config", "configs/x.yaml", "positional"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_unknown_command_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["no-such-command"]) != 0
    assert "invalid choice" in capsys.readouterr().err


def test_group_without_subcommand_exits_nonzero() -> None:
    assert main(["data"]) != 0


def test_implemented_command_rejects_unknown_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor", "--bogus"]) == 2
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


def test_qf_error_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing(args: object) -> int:
        raise QFError("boom")

    monkeypatch.setitem(
        COMMANDS, "doctor", dataclasses.replace(COMMANDS["doctor"], handler=failing)
    )
    assert main(["doctor"]) == 1
    assert "qf: error: boom" in capsys.readouterr().err


def test_invalid_log_level_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--log-level", "LOUD", "doctor"]) == 1
    assert "unknown log level" in capsys.readouterr().err


def test_string_system_exit_maps_to_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def exits(args: object) -> int:
        raise SystemExit("fatal: something")

    monkeypatch.setitem(COMMANDS, "doctor", dataclasses.replace(COMMANDS["doctor"], handler=exits))
    assert main(["doctor"]) == 1
    assert "fatal: something" in capsys.readouterr().err


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "qf", "export", "merge"], capture_output=True, text=True,
        check=False,
    )  # fmt: skip
    assert result.returncode == 2
    assert "Stage 16 is not implemented yet" in result.stderr


def test_system_exit_without_code_maps_to_0(monkeypatch: pytest.MonkeyPatch) -> None:
    def exits(args: object) -> int:
        raise SystemExit()

    monkeypatch.setitem(COMMANDS, "doctor", dataclasses.replace(COMMANDS["doctor"], handler=exits))
    assert main(["doctor"]) == 0
