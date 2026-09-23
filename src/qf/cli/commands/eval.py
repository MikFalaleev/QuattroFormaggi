"""`qf eval …` commands: evaluation of a backend on the frozen benchmark (step 9)."""

from __future__ import annotations

import argparse
from pathlib import Path

from qf.backends import BACKENDS
from qf.cli.wiring import build
from qf.common import load_yaml_config, project_root
from qf.eval import (
    FAKE_BANNER,
    REPORT_FILE,
    EvalConfig,
    compare_stage,
    load_bench,
    run_eval,
)

__all__ = ["configure_compare", "configure_run", "run_compare", "run_run"]


def configure_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="eval config (YAML)")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="continue an interrupted run: cases already answered are not asked again",
    )


def run_run(args: argparse.Namespace) -> int:
    root = project_root()
    cfg = load_yaml_config(args.config, EvalConfig)
    bench, records = load_bench(cfg, root)
    backend = build(BACKENDS, cfg.backend)
    run = run_eval(records, backend, cfg, bench=bench, root=root, resume=args.resume)
    if run.state["backend"] == "fake":
        print(FAKE_BANNER.removeprefix("> ").replace("**", ""))
    failed = sum(s.failed for s in run.scores)
    print(f"Backend {run.state['backend']}, model {run.state['model_id']}: "
          f"{len(run.scores)} cases, {failed} failed")  # fmt: skip
    for name, entry in run.table["overall"].items():
        value = entry["value"]
        print(f"  {name}: {'—' if value is None else round(value, 4)}")
    print(f"Report: {run.run_dir / REPORT_FILE}")
    return 0


def configure_compare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_a", type=Path, help="run directory A (the reference)")
    parser.add_argument("run_b", type=Path, help="run directory B (compared with A)")


def run_compare(args: argparse.Namespace) -> int:
    path, _ = compare_stage(args.run_a, args.run_b, root=project_root())
    print(f"Comparison: {path}")
    return 0
