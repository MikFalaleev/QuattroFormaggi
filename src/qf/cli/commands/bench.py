"""`qf bench …` commands: the frozen benchmark (step 8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qf.common import CONFIGS, load_yaml_config, project_root
from qf.data import (
    BenchmarkConfig,
    export_review,
    freeze,
    import_review_stage,
    load_manual_cases,
    verify_benchmark,
)

__all__ = [
    "DEFAULT_BENCH_CONFIG",
    "configure",
    "configure_freeze",
    "configure_import",
    "run_export",
    "run_freeze",
    "run_import",
    "run_verify",
]

DEFAULT_BENCH_CONFIG = CONFIGS / "eval" / "benchmark_v1.yaml"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"benchmark config (default: <project>/{DEFAULT_BENCH_CONFIG})",
    )


def _load(args: argparse.Namespace) -> tuple[Path, BenchmarkConfig]:
    root = project_root()
    return root, load_yaml_config(args.config or root / DEFAULT_BENCH_CONFIG, BenchmarkConfig)


def run_export(args: argparse.Namespace) -> int:
    root, cfg = _load(args)
    outcome = export_review(cfg, root=root)
    for spec in cfg.slices:
        main = sum(c.slice == spec.name and c.role == "main" for c in outcome.candidates)
        spare = sum(c.slice == spec.name and c.role == "reserve" for c in outcome.candidates)
        print(f"{spec.name}: {main} main + {spare} reserve")
    print(f"Candidates to review: {len(outcome.candidates)}")
    print(f"Read: {outcome.markdown_path}")
    print(f"Fill in the verdict column (ok / fix / drop): {outcome.csv_path}")
    print(f"Run manifest: {outcome.run_dir}")
    return 0


def configure_import(parser: argparse.ArgumentParser) -> None:
    configure(parser)
    parser.add_argument("--csv", type=Path, default=None, help="review CSV with verdicts")


def run_import(args: argparse.Namespace) -> int:
    root, cfg = _load(args)
    result = import_review_stage(cfg, args.csv or root / cfg.review_csv, root=root)
    ok = len(result.kept) - len(result.unreviewed)
    print(f"ok {ok}, fix {len(result.fixes)}, drop {len(result.drops)}, "
          f"no verdict {len(result.unreviewed)}")  # fmt: skip
    if result.fixes:
        print(f"Records for generator fixes: {cfg.work_dir / 'fix_list.md'}")
    return 0


def configure_freeze(parser: argparse.ArgumentParser) -> None:
    configure(parser)
    parser.add_argument("--manual", type=Path, default=None, help="YAML with hand-written cases")
    parser.add_argument(
        "--allow-unreviewed", action="store_true", help="freeze records without a verdict (tests)"
    )


def run_freeze(args: argparse.Namespace) -> int:
    root, cfg = _load(args)
    manual, notes = (load_manual_cases(args.manual, cfg.schema_version) if args.manual
                     else ([], []))  # fmt: skip
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    outcome = freeze(cfg, root=root, manual=manual, manual_source=args.manual,
                     allow_unreviewed=args.allow_unreviewed)  # fmt: skip
    for name, count in outcome.counts.items():
        short = outcome.shortfalls.get(name)
        print(f"{name}: {count}" + (f" (short of target by {short})" if short else ""))
    print(f"manual: {outcome.manual}")
    for warning in outcome.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Frozen: {outcome.bench.path} (sha256 {outcome.bench.sha256})")
    print(f"Splits without the benchmark: {cfg.out_dir}/")
    print(f"Run manifest: {outcome.run_dir}")
    return 0


def run_verify(args: argparse.Namespace) -> int:
    root, cfg = _load(args)
    ref = verify_benchmark(root / cfg.bench_path, root=root)
    print(f"OK: {ref.path} matches its .sha256 and manifest (sha256 {ref.sha256})")
    return 0
