"""`qf export adapter|verify` (plan step 15, D-121): a backup copy of the accepted adapter in
`artifacts/adapters/<name>/` and its check against the manifests. Merge and GGUF: steps 16–17."""

from __future__ import annotations

import argparse
from pathlib import Path

from qf.common import load_yaml_config, project_root
from qf.contracts import ModelRef
from qf.export import ADAPTERS_DIR, backup_adapter, verify_adapter
from qf.training import BaseModelConfig

BASE_MODEL_CONFIG = Path("configs/train/base_model.yaml")


def _pinned_base(root: Path) -> ModelRef:
    base = load_yaml_config(root / BASE_MODEL_CONFIG, BaseModelConfig)
    return ModelRef(id=base.id, revision=base.revision)


def _adapter_dir(path: Path, root: Path) -> Path:
    """A run directory or the adapter directory itself, relative to the project root."""
    absolute = path if path.is_absolute() else root / path
    if (absolute / "adapter").is_dir() and not (absolute / "adapter_config.json").exists():
        absolute = absolute / "adapter"
    return absolute.relative_to(root) if absolute.is_relative_to(root) else absolute


def _print(check: object) -> None:
    for line in getattr(check, "checks", []):
        print(f"  ok: {line}")
    for line in getattr(check, "problems", []):
        print(f"  PROBLEM: {line}")


def configure_adapter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="runs/<id> of the accepted training run "
                        "(or its adapter directory)")  # fmt: skip
    parser.add_argument("--name", required=True,
                        help=f"backup name: {ADAPTERS_DIR.as_posix()}/<name>/")  # fmt: skip


def run_adapter(args: argparse.Namespace) -> int:
    root = project_root()
    source = _adapter_dir(args.source, root)
    before = verify_adapter(source, root, base=_pinned_base(root))
    _print(before)
    if not before.ok:
        print(f"{source}: not copied — the source adapter is not intact")
        return 1
    ref = backup_adapter(source, args.name, root)
    after = verify_adapter(Path(ref.path), root, base=_pinned_base(root))
    _print(after)
    print(f"Backup: {ref.path} (sha256 {ref.sha256[:12]}…, manifest {ref.path.name}.manifest.json)")
    print("⛔ Now copy this directory and its .manifest.json OFF this machine yourself "
          "(external disk, your cloud); the agent uploads nothing. Check a restored copy with "
          f"`qf export verify --adapter {ref.path}`.")  # fmt: skip
    return 0 if after.ok else 1


def configure_verify(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", type=Path, required=True,
                        help="an adapter directory (artifacts/adapters/<name> or "
                        "runs/<id>/adapter) or a training run")  # fmt: skip


def run_verify(args: argparse.Namespace) -> int:
    root = project_root()
    adapter = _adapter_dir(args.adapter, root)
    check = verify_adapter(adapter, root, base=_pinned_base(root))
    _print(check)
    print(f"{adapter}: {'intact' if check.ok else 'NOT intact'}")
    return 0 if check.ok else 1
