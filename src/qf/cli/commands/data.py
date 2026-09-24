"""`qf data …` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from qf.cli.wiring import build
from qf.common import (
    CONFIGS,
    DATA_PROCESSED,
    NotImplementedStageError,
    QFError,
    load_yaml_config,
    project_root,
    read_artifact,
)
from qf.contracts import supported_versions
from qf.data import (
    DATASET_FILES,
    FACTS_BUILDERS,
    FACTS_V2_REPORT_FILENAME,
    LOAD_FACTS_FILENAME,
    LOAD_FACTS_V2_FILENAME,
    RAW_SOURCES,
    SPLIT_FILES,
    SPLITTERS,
    Expectations,
    FactsFileConfig,
    GenerateConfig,
    SourceFileConfig,
    SplitFileConfig,
    build_facts_artifact,
    build_facts_v2_artifact,
    build_length_report,
    fetch_raw_dataset,
    generate_dataset,
    profile_raw_dataset,
    raw_dataset_path,
    read_provenance,
    read_sft_records,
    run_validation,
    split_dataset,
    split_issues,
    verify_raw_dataset,
)

__all__ = [
    "DEFAULT_CONDITIONS_TABLE",
    "DEFAULT_EXPECTATIONS",
    "DEFAULT_FACTS_CONFIG",
    "DEFAULT_DATA_DIR",
    "DEFAULT_GENERATE_CONFIG",
    "DEFAULT_SPLIT_CONFIG",
    "DEFAULT_SOURCE_CONFIG",
    "configure_build",
    "configure_facts",
    "configure_facts_v2",
    "configure_fetch",
    "configure_profile",
    "configure_report",
    "configure_split",
    "configure_validate",
    "run_build",
    "run_facts",
    "run_facts_v2",
    "run_fetch",
    "run_profile",
    "run_report",
    "run_split",
    "run_validate",
]

DEFAULT_SOURCE_CONFIG = CONFIGS / "data" / "source.yaml"
DEFAULT_EXPECTATIONS = CONFIGS / "data" / "expectations.yaml"
DEFAULT_FACTS_CONFIG = CONFIGS / "data" / "facts.yaml"
DEFAULT_CONDITIONS_TABLE = CONFIGS / "data" / "equipment_conditions_v1.yaml"
DEFAULT_GENERATE_CONFIG = CONFIGS / "data" / "generate_v1.yaml"
DEFAULT_SPLIT_CONFIG = CONFIGS / "data" / "split.yaml"
DEFAULT_DATA_DIR = DATA_PROCESSED / "generated_v1"


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


def configure_facts_v2(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=f"equipment and conditions table (default: <project>/{DEFAULT_CONDITIONS_TABLE})",
    )


def run_facts_v2(args: argparse.Namespace) -> int:
    """Sub-step V2: load facts v1 + the table -> load facts v2 and a report to review."""
    root = project_root()
    result = build_facts_v2_artifact(
        root / DATA_PROCESSED / LOAD_FACTS_FILENAME,
        args.table or root / DEFAULT_CONDITIONS_TABLE,
        root=root,
        out_path=root / DATA_PROCESSED / LOAD_FACTS_V2_FILENAME,
        report_path=root / DATA_PROCESSED / FACTS_V2_REPORT_FILENAME,
    )
    print(f"Built {result.count} load facts v2: {result.ref.path} "
          f"(sha256 {result.ref.sha256[:12]})")  # fmt: skip
    print(f"Review: {result.report_path}")
    print(f"Run manifest: {result.run_dir}")
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


def _facts_v1(root: Path, facts_v1: Path) -> Path:
    return facts_v1


def _facts_v2(root: Path, facts_v1: Path) -> Path:
    """Sub-step V2 with the default table: the facts card_v2 is generated from."""
    result = build_facts_v2_artifact(
        facts_v1, root / DEFAULT_CONDITIONS_TABLE, root=root,
        out_path=root / DATA_PROCESSED / LOAD_FACTS_V2_FILENAME,
        report_path=root / DATA_PROCESSED / FACTS_V2_REPORT_FILENAME,
    )  # fmt: skip
    print(f"Built {result.count} load facts v2: {result.ref.path} "
          f"(sha256 {result.ref.sha256[:12]})")  # fmt: skip
    return root / result.ref.path


_FACTS_FOR_SCHEMA = {"card_v1": _facts_v1, "card_v2": _facts_v2}
"""The facts the generator of each answer schema reads (card_v2: derived from card_v1 facts)."""


def run_build(args: argparse.Namespace) -> int:
    """Facts (step 5; for card_v2 also sub-step V2), then the SFT records generated from them
    (step 6; card_v2: sub-step V3)."""
    root = project_root()
    config_path = args.config or root / DEFAULT_GENERATE_CONFIG
    cfg = load_yaml_config(config_path, GenerateConfig)
    facts_path = _FACTS_FOR_SCHEMA[cfg.schema_version](
        root, _build_facts(root, args.facts_config, args.source_config)
    )
    source, _ = _load_source(root, args.source_config)
    provenance = read_provenance(raw_dataset_path(source, root))
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


# --- step 7: validation, splits, length report ---------------------------------------------


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"directory with the split files (default: <project>/{DEFAULT_DATA_DIR})",
    )


def _dataset_paths(root: Path, data_dir: Path | None) -> dict[str, Path]:
    """Split files (all required) and smoke (if present) of a dataset directory."""
    base = data_dir if data_dir is not None else root / DEFAULT_DATA_DIR
    base = base if base.is_absolute() else root / base
    missing = [name for name in SPLIT_FILES if not (base / f"{name}.jsonl").is_file()]
    if missing:
        raise QFError(f"{base}: missing split files {', '.join(missing)} (run `qf data build`)")
    return {n: base / f"{n}.jsonl" for n in DATASET_FILES if (base / f"{n}.jsonl").is_file()}


def configure_validate(parser: argparse.ArgumentParser) -> None:
    _add_data_dir(parser)
    parser.add_argument("--bench", type=Path, default=None, help="benchmark file to check too")


def run_validate(args: argparse.Namespace) -> int:
    root = project_root()
    paths = _dataset_paths(root, args.data_dir)
    out_path = next(iter(paths.values())).parent / "validation_report.json"
    outcome = run_validation(paths, args.bench, root=root, out_path=out_path)
    report = outcome.report
    for name, info in report.datasets.items():
        print(f"{name}: {info['records']} records")
    for code, count in report.counts.items():
        kind = "warning" if any(w.code == code for w in report.warnings) else "error"
        print(f"{kind:7}  {code}: {count}")
    for item in report.errors[:10]:
        print(f"  {item.record_id} [{item.split}] {item.code}: {item.message}", file=sys.stderr)
    print(f"Report: {outcome.ref.path}")
    print(f"Run manifest: {outcome.run_dir}")
    if not report.passed:
        print(f"qf: validation failed: {len(report.errors)} error(s)", file=sys.stderr)
        return 1
    print(f"OK: 0 errors, {len(report.warnings)} warning(s)")
    return 0


def configure_split(parser: argparse.ArgumentParser) -> None:
    _add_data_dir(parser)
    parser.add_argument(
        "--input", type=Path, default=None, help="dataset without splits: assign them by group"
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write the splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"splitter config (default: <project>/{DEFAULT_SPLIT_CONFIG})",
    )


def run_split(args: argparse.Namespace) -> int:
    """With --input: assign splits by group. Without: check the splits of --data-dir."""
    root = project_root()
    if args.input is None:
        datasets = {}
        for name, path in _dataset_paths(root, args.data_dir).items():
            if name in SPLIT_FILES:
                versions = supported_versions("sft_dataset")
                ref = read_artifact(path, "sft_dataset", versions, root=root)
                datasets[name] = read_sft_records(root / ref.path, name)[0]
        issues = split_issues(datasets)
        for issue in issues[:10]:
            line = f"  {issue.record_id} [{issue.split}] {issue.code}: {issue.message}"
            print(line, file=sys.stderr)
        if issues:
            print(f"qf: {len(issues)} split problem(s)", file=sys.stderr)
            return 1
        counts = ", ".join(f"{name} {len(items)}" for name, items in datasets.items())
        print(f"OK: splits already assigned, no group in two splits ({counts})")
        return 0
    if args.out_dir is None:
        raise QFError("--input needs --out-dir")
    file_config = load_yaml_config(args.config or root / DEFAULT_SPLIT_CONFIG, SplitFileConfig)
    result = split_dataset(
        build(SPLITTERS, file_config.splitter), args.input, root=root,
        out_dir=args.out_dir if args.out_dir.is_absolute() else root / args.out_dir,
        config=file_config.model_dump(mode="json"),
    )  # fmt: skip
    for name, ref in result.refs.items():
        print(f"{name}: {result.counts[name]} records -> {ref.path}")
    print(f"Run manifest: {result.run_dir}")
    return 0


def configure_report(parser: argparse.ArgumentParser) -> None:
    _add_data_dir(parser)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="tokenizer directory for token lengths (step 11)")  # fmt: skip


def run_report(args: argparse.Namespace) -> int:
    if args.tokenizer is not None:
        raise NotImplementedStageError("11", "qf data report --tokenizer")
    root = project_root()
    paths = _dataset_paths(root, args.data_dir)
    outcome = build_length_report(paths, root=root, out_dir=next(iter(paths.values())).parent)
    for name, count in outcome.report["records"].items():
        user = outcome.report["chars"][name]["user"]
        print(f"{name}: {count} records, user chars p50 {user['p50']}, max {user['max']}")
    print(f"Report: {outcome.ref.path} (and length_report.md next to it)")
    print(f"Run manifest: {outcome.run_dir}")
    return 0
