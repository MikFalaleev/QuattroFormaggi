"""`qf export adapter|verify|merge` (plan steps 15–16, D-121, D-122): a backup copy of the
accepted adapter in `artifacts/adapters/<name>/`, its check against the manifests, the merge
into full weights in `artifacts/merged/<name>/` and the check of the merged model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qf.cli.wiring import build
from qf.common import (
    QFError,
    atomic_write_text,
    collect_hardware,
    load_yaml_config,
    project_root,
    read_artifact,
    start_run,
)
from qf.contracts import ModelRef, supported_versions
from qf.domain import parse_sft_jsonl
from qf.export import (
    ADAPTERS_DIR,
    EXPORT_MANIFEST,
    MERGERS,
    ExportMergeConfig,
    MergeTarget,
    backup_adapter,
    verify_adapter,
    verify_merged,
    verify_merged_files,
)
from qf.training import BaseModelConfig

BASE_MODEL_CONFIG = Path("configs/train/base_model.yaml")
MERGE_CONFIG = Path("configs/export/merge.yaml")
MERGED_DIR = Path("artifacts/merged/Quattro-Formaggi-12B-Logistics-v0.1")
VERIFY_FILE = "verify_merged.json"
PACKAGES = ("torch", "transformers", "peft", "safetensors", "accelerate")


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
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--adapter", type=Path,
                      help="an adapter directory (artifacts/adapters/<name> or "
                      "runs/<id>/adapter) or a training run: files only")  # fmt: skip
    what.add_argument("--merged", type=Path,
                      help="a merged model of `qf export merge` (step 16): with the models, "
                      "or only its files with --files-only")  # fmt: skip
    parser.add_argument("--files-only", action="store_true",
                        help="--merged: hashes only, no torch (the copy on the Mac)")  # fmt: skip
    parser.add_argument("--config", type=Path, default=MERGE_CONFIG,
                        help="--merged: the merger (base, dtype, device), prompts")  # fmt: skip


def run_verify(args: argparse.Namespace) -> int:
    root = project_root()
    if args.merged is not None:
        return _verify_merged(args, root)
    adapter = _adapter_dir(args.adapter, root)
    check = verify_adapter(adapter, root, base=_pinned_base(root))
    _print(check)
    print(f"{adapter}: {'intact' if check.ok else 'NOT intact'}")
    return 0 if check.ok else 1


def _merge_config(path: Path, root: Path) -> tuple[ExportMergeConfig, MergeTarget, ModelRef]:
    cfg = load_yaml_config(path if path.is_absolute() else root / path, ExportMergeConfig)
    target = MergeTarget.of(cfg.merger)
    base = ModelRef(id=target.base_id, revision=target.revision)
    pinned = _pinned_base(root)
    if base != pinned:
        raise QFError(f"{path}: base {base.id}@{base.revision} is not the pinned "
                      f"{pinned.id}@{pinned.revision} ({BASE_MODEL_CONFIG})")  # fmt: skip
    return cfg, target, base


def configure_merge(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", type=Path, required=True,
                        help="the accepted adapter: artifacts/adapters/<name>")  # fmt: skip
    parser.add_argument("--out", type=Path, default=MERGED_DIR,
                        help=f"the merged model directory (default {MERGED_DIR})")  # fmt: skip
    parser.add_argument("--config", type=Path, default=MERGE_CONFIG,
                        help="the merger and its parameters")  # fmt: skip


def run_merge(args: argparse.Namespace) -> int:
    """Base (full precision) + adapter -> `hf_model` artifact, with a run manifest."""
    root = project_root()
    cfg, target, base = _merge_config(args.config, root)
    adapter = read_artifact(_adapter_dir(args.adapter, root), "lora_adapter",
                            supported_versions("lora_adapter"), root=root)  # fmt: skip
    merger = build(MERGERS, cfg.merger)
    run = start_run("export-merge", root)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    ref = merger.merge(base, adapter, args.out, run_id=run.run_id)
    manifest = json.loads((root / ref.path / EXPORT_MANIFEST).read_text(encoding="utf-8"))
    run.finish(
        packages=PACKAGES, hardware=collect_hardware(), base_revision=base.revision,
        config={**cfg.model_dump(mode="json"), "adapter": adapter.path.as_posix(),
                "out": ref.path.as_posix()},
        data_hashes={adapter.path.as_posix(): adapter.sha256, ref.path.as_posix(): ref.sha256},
        metrics={"files": len(manifest["files"]), "dtype": target.dtype},
    )  # fmt: skip
    print(f"Merged: {ref.path} (sha256 {ref.sha256[:12]}…, {len(manifest['files'])} files, "
          f"{target.dtype}); run {run.run_id}")  # fmt: skip
    print(f"Next: `qf export verify --merged {ref.path}` (the models, same machine).")
    return 0


def _prompts(bench: Path, n: int, root: Path) -> list[list[dict[str, str]]]:
    """System + user messages of the first `n` records (never the gold answer)."""
    records, issues = parse_sft_jsonl((root / bench).read_text(encoding="utf-8"),
                                      source=bench.name)  # fmt: skip
    if issues or len(records) < n:
        raise QFError(f"{bench}: {len(issues)} invalid records, {len(records)} valid (need {n})")
    return [[{"role": m.role, "content": m.content} for m in r.messages[:2]]
            for r in records[:n]]  # fmt: skip


def _verify_merged(args: argparse.Namespace, root: Path) -> int:
    merged = args.merged if not args.merged.is_absolute() else args.merged.relative_to(root)
    cfg, target, _ = _merge_config(args.config, root)
    if args.files_only:
        check = verify_merged_files(merged, root, base_dir=root / target.base_dir)
        _print(check)
        print(f"{merged}: {'intact' if check.ok else 'NOT intact'} (files)")
        return 0 if check.ok else 1
    own: dict[str, Any] = json.loads((root / merged / EXPORT_MANIFEST).read_text(encoding="utf-8"))
    adapter = Path(own["adapter"]["path"])
    run = start_run("export-verify", root)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    check = verify_merged(merged, adapter, target.base_dir, root,
                          messages=_prompts(cfg.verify.bench, cfg.verify.prompts, root),
                          max_new_tokens=cfg.verify.max_new_tokens, dtype=target.dtype,
                          device=target.device,
                          fix_mistral_regex=target.fix_mistral_regex)  # fmt: skip
    atomic_write_text(run.run_dir / VERIFY_FILE, json.dumps(check.as_dict(), indent=2) + "\n")
    hashes = {merged.as_posix(): check.sha256 or "", adapter.as_posix(): own["adapter"]["sha256"]}
    run.finish(
        packages=PACKAGES, hardware=collect_hardware(),
        status="completed" if check.ok else "failed", base_revision=target.revision,
        config={**cfg.model_dump(mode="json"), "merged": merged.as_posix()}, data_hashes=hashes,
        metrics={"ok": check.ok, "prompts": len(check.prompts),
                 "greedy_equal": sum(p.greedy_equal for p in check.prompts),
                 "differs_from_base": sum(p.differs_from_base for p in check.prompts),
                 "max_abs_diff": max((p.max_abs_diff for p in check.prompts), default=None)},
    )  # fmt: skip
    _print(check)
    print(f"{merged}: {'verified' if check.ok else 'NOT verified'}; report "
          f"{run.run_dir.relative_to(root) / VERIFY_FILE}")  # fmt: skip
    return 0 if check.ok else 1
