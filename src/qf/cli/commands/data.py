"""`qf data …` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qf.cli.wiring import build
from qf.common import CONFIGS, load_yaml_config, project_root
from qf.data import RAW_SOURCES, SourceFileConfig, fetch_raw_dataset, verify_raw_dataset

__all__ = ["DEFAULT_SOURCE_CONFIG", "configure_fetch", "run_fetch"]

DEFAULT_SOURCE_CONFIG = CONFIGS / "data" / "source.yaml"


def configure_fetch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"source config (default: <project>/{DEFAULT_SOURCE_CONFIG})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not download; check the fetched files against their provenance",
    )


def run_fetch(args: argparse.Namespace) -> int:
    root = project_root()
    config_path: Path = args.config if args.config is not None else root / DEFAULT_SOURCE_CONFIG
    file_config = load_yaml_config(config_path, SourceFileConfig)
    source = build(RAW_SOURCES, file_config.source)
    if args.verify_only:
        ref, problems = verify_raw_dataset(source, root=root)
        if problems:
            print(f"{ref.path} does not match its provenance:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"OK: {ref.path} matches its provenance (sha256 {ref.sha256[:12]})")
        return 0
    result = fetch_raw_dataset(
        source, root=root, source_config=file_config.source.model_dump(mode="json")
    )
    if result.downloaded:
        print(f"Downloaded {result.ref.path} (sha256 {result.ref.sha256[:12]})")
        print(f"Run manifest: {result.run_dir}")
    else:
        print(f"Already present and verified: {result.ref.path}; nothing downloaded")
    return 0
