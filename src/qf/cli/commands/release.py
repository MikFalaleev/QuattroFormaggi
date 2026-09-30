"""`qf release check|build`: the release of v0.1 (step 19). Nothing is uploaded."""

from __future__ import annotations

import argparse
from pathlib import Path

from qf.common import load_yaml_config, project_root
from qf.export import RELEASE_MANIFEST, ReleaseConfig, build_release, release_check

__all__ = ["configure_build", "configure_check", "run_build", "run_check"]

RELEASE_DIR = Path("artifacts/release")
RELEASE_CONFIG = Path("configs/release/release_v0.1.yaml")


def configure_build(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=RELEASE_CONFIG,
                        help=f"what goes into the release (default: {RELEASE_CONFIG})")  # fmt: skip


def run_build(args: argparse.Namespace) -> int:
    """Copy the cards into the release directory and write `release_manifest.json` from the
    artifacts' manifests. Nothing is uploaded; run `qf release check` afterwards."""
    root = project_root()
    cfg = load_yaml_config(args.config if args.config.is_absolute() else root / args.config,
                           ReleaseConfig)  # fmt: skip
    release_dir = cfg.release_dir if cfg.release_dir.is_absolute() else root / cfg.release_dir
    manifest = build_release(
        release_dir, release=cfg.release, license_id=cfg.license, base_model=cfg.base_model,
        base_revision=cfg.base_revision, artifacts={k: root / v for k, v in cfg.artifacts.items()},
        card_sources={k: root / v for k, v in cfg.card_sources.items()}, docs=cfg.docs, root=root,
    )  # fmt: skip
    print(f"{release_dir / RELEASE_MANIFEST}: {len(manifest.artifacts)} artifacts, "
          f"{len(manifest.cards)} cards; run `qf release check`")  # fmt: skip
    return 0


def configure_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR,
                        help=f"release directory (default: {RELEASE_DIR})")  # fmt: skip
    parser.add_argument("--skip-hashes", action="store_true",
                        help="compare manifests only, do not hash large artifacts")  # fmt: skip


def run_check(args: argparse.Namespace) -> int:
    root = project_root()
    release_dir = args.release_dir if args.release_dir.is_absolute() else root / args.release_dir
    problems = release_check(release_dir, root=root, verify_hashes=not args.skip_hashes)
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    mode = "manifests only, content hashes NOT verified" if args.skip_hashes else "hashes verified"
    if problems:
        print(
            f"{release_dir / RELEASE_MANIFEST}: NOT complete ({len(problems)} problem(s); {mode})"
        )
        return 1
    print(f"{release_dir / RELEASE_MANIFEST}: complete ({mode})")
    return 0
