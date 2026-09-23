"""`qf` command-line entry point.

Every planned command is registered up front. A command whose plan step is not implemented
exits with code 2 and says so (rule A.2.2); it never fakes success.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qf import __version__
from qf.cli.commands import data, doctor
from qf.common import NotImplementedStageError, QFError, setup_logging

__all__ = ["COMMANDS", "EXIT_ERROR", "EXIT_NOT_IMPLEMENTED", "EXIT_OK", "CommandSpec", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2

Handler = Callable[[argparse.Namespace], int]
Configure = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandSpec:
    name: str  # "doctor" or "<group> <command>"; a name is never both a command and a group
    stage: str  # IMPLEMENTATION_PLAN.md step that implements the command
    help: str
    handler: Handler | None = None
    configure: Configure | None = None

    @property
    def implemented(self) -> bool:
        return self.handler is not None


_SPECS = (
    CommandSpec("doctor", "1", "read-only environment report", doctor.run, doctor.configure),
    CommandSpec(
        "data fetch", "2", "download the pinned raw dataset", data.run_fetch, data.configure_fetch
    ),
    CommandSpec(
        "data profile",
        "3",
        "check raw table integrity",
        data.run_profile,
        data.configure_profile,
    ),
    CommandSpec(
        "data facts",
        "5",
        "build load facts from the raw dataset",
        data.run_facts,
        data.configure_facts,
    ),
    CommandSpec(
        "data build",
        "6",
        "build load facts and generate SFT records",
        data.run_build,
        data.configure_build,
    ),
    CommandSpec("data split", "7", "assign or check splits", data.run_split, data.configure_split),
    CommandSpec(
        "data report",
        "7",
        "dataset length and composition report",
        data.run_report,
        data.configure_report,
    ),
    CommandSpec(
        "validate-data",
        "7",
        "validate datasets: schema, leakage, duplicates",
        data.run_validate,
        data.configure_validate,
    ),
    CommandSpec("bench export-review", "8", "export benchmark for human review"),
    CommandSpec("bench import-review", "8", "import human review verdicts"),
    CommandSpec("bench freeze", "8", "freeze the reviewed benchmark"),
    CommandSpec("bench verify", "8", "verify the frozen benchmark hash"),
    CommandSpec("eval run", "9", "run an evaluation from a config"),
    CommandSpec("eval compare", "9", "paired comparison of two eval runs"),
    CommandSpec("eval-baseline", "10", "baseline of the base model in LM Studio"),
    CommandSpec("tokens audit", "11", "audit tokenizer, chat template and loss mask"),
    CommandSpec("train estimate", "12", "estimate steps, time and cost of training"),
    CommandSpec("train run", "12", "run QLoRA adapter training"),
    CommandSpec("export verify", "15", "verify adapter/merged artifacts against manifests"),
    CommandSpec("export merge", "16", "merge the adapter into full HF weights"),
    CommandSpec("export gguf", "17", "convert to GGUF and quantize"),
    CommandSpec("extract", "18", "extract a shipment card from a request"),
    CommandSpec("release check", "19", "check release completeness"),
    CommandSpec("lab", "L1-L6", "QF-Lab from-scratch Transformer track"),
)
COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in _SPECS}


def _help_text(spec: CommandSpec) -> str:
    status = "" if spec.implemented else ", not implemented yet"
    return f"{spec.help} [step {spec.stage}{status}]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qf", description="Quattro Formaggi pipeline CLI")
    parser.add_argument("--version", action="version", version=f"qf {__version__}")
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    top = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    groups: dict[str, argparse._SubParsersAction[argparse.ArgumentParser]] = {}
    for spec in COMMANDS.values():
        head, *tail = spec.name.split()
        if not tail:
            sub = top.add_parser(head, help=_help_text(spec))
        else:
            if head not in groups:
                group_parser = top.add_parser(head, help=f"{head} commands")
                groups[head] = group_parser.add_subparsers(
                    dest=f"{head}_command", metavar="<subcommand>", required=True
                )
            sub = groups[head].add_parser(tail[0], help=_help_text(spec))
        sub.set_defaults(spec=spec)
        if spec.configure is not None:
            spec.configure(sub)
    return parser


def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return EXIT_OK
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args, unknown = parser.parse_known_args(argv)
    except SystemExit as exc:  # --version, --help and usage errors
        return _exit_code(exc)
    spec: CommandSpec = args.spec
    try:
        if spec.handler is None:
            raise NotImplementedStageError(spec.stage, spec.name)
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        setup_logging(args.log_level)
        return spec.handler(args)
    except SystemExit as exc:
        return _exit_code(exc)
    except NotImplementedStageError as exc:
        print(f"qf: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except QFError as exc:
        print(f"qf: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
