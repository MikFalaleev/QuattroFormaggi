"""`qf doctor`: read-only environment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qf.common import (
    DEFAULT_REPORT_PATH,
    atomic_write_text,
    collect_environment,
    format_environment_report,
    project_root,
)

__all__ = ["configure", "run"]


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to save the JSON report (default: <project>/{DEFAULT_REPORT_PATH})",
    )


def run(args: argparse.Namespace) -> int:
    out: Path = args.out if args.out is not None else project_root() / DEFAULT_REPORT_PATH
    env = collect_environment()
    atomic_write_text(out, json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(format_environment_report(env))
    print(f"Report saved to {out}")
    return 0
