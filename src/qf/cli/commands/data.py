"""`qf data …` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qf.cli.wiring import build
from qf.common import CONFIGS, DATA_PROCESSED, load_yaml_config, project_root
from qf.data import (
    FACTS_BUILDERS,
    LOAD_FACTS_FILENAME,
    RAW_SOURCES,
    Expectations,
    FactsFileConfig,
    GenerateConfig,
    SourceFileConfig,
    build_facts_artifact,
    fetch_raw_dataset,
    generate_dataset,
    profile_raw_dataset,
    raw_dataset_path,
    read_provenance,
    verify_raw_dataset,
)

__all__ = [
    "DEFAULT_EXPECTATIONS",
    "DEFAULT_FACTS_CONFIG",
    "DEFAULT_GENERATE_CONFIG",
    "DEFAULT_SOURCE_CONFIG",
    "configure_build",
    "configure_facts",
    "configure_fetch",
    "configure_profile",
    "run_build",
    "run_facts",
    "run_fetch",
    "run_profile",
]

DEFAULT_SOURCE_CONFIG = CONFIGS / "data" / "source.yaml"
DEFAULT_EXPECTATIONS = CONFIGS / "data" / "expectations.yaml"
DEFAULT_FACTS_CONFIG = CONFIGS / "data" / "facts.yaml"
DEFAULT_GENERATE_CONFIG = CONFIGS / "data" / "generate_v1.yaml"


def _add_source_config(parser: argparse.ArgumentParser, flag: str) -> None:
    parser.add_argument(
        flag,
        dest="source_config",
        type=Path,
        default=None,
        help=f"source config (default: <project>/{DEFAULT_SOURCE_CONFIG})",
    )


def _load_source(root: Path, config_path: Path | None) -> tuple[Any, SourceFileConfig]:
    path = config_path if config_path is not None else root / DEFAULT_SOURCE_CONFIG
    file_config = load_yaml_config(path, SourceFileConfig)
    return build(RAW_SOURCES, file_config.source), file_config


def configure_fetch(parser: argparse.ArgumentParser) -> None:
    _add_source_config(parser, "--config")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not download; check the fetched files against their provenance",
    )


def run_fetch(args: argparse.Namespace) -> int:
    root = project_root()
    source, file_config = _load_source(root, args.source_config)
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


def configure_profile(parser: argparse.ArgumentParser) -> None:
    _add_source_config(parser, "--source-config")
    parser.add_argument(
        "--expectations",
        type=Path,
        default=None,
        help=f"expected table properties (default: <project>/{DEFAULT_EXPECTATIONS})",
    )


def run_profile(args: argparse.Namespace) -> int:
    root = project_root()
    source, _ = _load_source(root, args.source_config)
    expectations_path = args.expectations or root / DEFAULT_EXPECTATIONS
    expectations = load_yaml_config(expectations_path, Expectations)
    outcome = profile_raw_dataset(
        raw_dataset_path(source, root),
        expectations,
        root=root,
        out_dir=root / DATA_PROCESSED,
        expectations_config=expectations.model_dump(mode="json"),
    )
    for check in outcome.report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}  {check.name}")
        for detail in check.details if not check.passed else []:
            print(f"      {detail}")
    print(f"Report: {outcome.json_path} and {outcome.markdown_path}")
    print(f"Run manifest: {outcome.run_dir}")
    if not outcome.report.passed:
        print(f"qf: profile failed: {', '.join(outcome.report.failed_checks)}", file=sys.stderr)
        return 1
    return 0


def configure_facts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"facts builder config (default: <project>/{DEFAULT_FACTS_CONFIG})",
    )
    _add_source_config(parser, "--source-config")


def _build_facts(root: Path, facts_config: Path | None, source_config: Path | None) -> Path:
    """The facts stage (step 5); returns the path of the load_facts artifact."""
    source, _ = _load_source(root, source_config)
    file_config = load_yaml_config(facts_config or root / DEFAULT_FACTS_CONFIG, FactsFileConfig)
    result = build_facts_artifact(
        build(FACTS_BUILDERS, file_config.builder),
        raw_dataset_path(source, root),
        root=root,
        out_path=root / DATA_PROCESSED / LOAD_FACTS_FILENAME,
        config=file_config.model_dump(mode="json"),
    )
    print(f"Built {result.count} load facts: {result.ref.path} (sha256 {result.ref.sha256[:12]})")
    print(f"Run manifest: {result.run_dir}")
    return root / result.ref.path


def run_facts(args: argparse.Namespace) -> int:
    _build_facts(project_root(), args.config, args.source_config)
    return 0


def configure_build(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"generator config (default: <project>/{DEFAULT_GENERATE_CONFIG})",
    )
    parser.add_argument(
        "--facts-config",
        type=Path,
        default=None,
        help=f"facts builder config (default: <project>/{DEFAULT_FACTS_CONFIG})",
    )
    _add_source_config(parser, "--source-config")


def run_build(args: argparse.Namespace) -> int:
    """Facts (step 5), then the SFT records generated from them (step 6)."""
    root = project_root()
    facts_path = _build_facts(root, args.facts_config, args.source_config)
    source, _ = _load_source(root, args.source_config)
    provenance = read_provenance(raw_dataset_path(source, root))
    config_path = args.config or root / DEFAULT_GENERATE_CONFIG
    cfg = load_yaml_config(config_path, GenerateConfig)
    result = generate_dataset(
        facts_path,
        cfg,
        root=root,
        out_dir=root / DATA_PROCESSED / cfg.dataset_name,
        source_prefix=f"{provenance.repo_id}@{provenance.revision[:12]}",
        config_path=config_path,
    )
    for name, ref in result.refs.items():
        print(f"{name}: {result.counts[name]} records -> {ref.path} (sha256 {ref.sha256[:12]})")
    print(f"Run manifest: {result.run_dir}")
    return 0
