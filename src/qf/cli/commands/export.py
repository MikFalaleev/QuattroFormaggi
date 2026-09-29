"""`qf export adapter|verify|merge|gguf` (plan steps 15–17, D-121…D-123): a backup copy of the
accepted adapter in `artifacts/adapters/<name>/`, its check against the manifests, the merge
into full weights in `artifacts/merged/<name>/`, the check of the merged model and the GGUF
files of the pinned llama.cpp in `artifacts/gguf/`."""

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
    load_artifact_manifest,
    load_yaml_config,
    project_root,
    read_artifact,
    sha256_file,
    start_run,
)
from qf.contracts import ModelRef, supported_versions
from qf.domain import parse_sft_jsonl
from qf.export import (
    ADAPTERS_DIR,
    CONVERTERS,
    EXPORT_MANIFEST,
    MERGERS,
    QUANTIZERS,
    ExportGgufConfig,
    ExportMergeConfig,
    MergeTarget,
    backup_adapter,
    check_gguf_against_hf,
    read_gguf_metadata,
    verify_adapter,
    verify_merged,
    verify_merged_files,
)
from qf.training import BaseModelConfig

BASE_MODEL_CONFIG = Path("configs/train/base_model.yaml")
MERGE_CONFIG = Path("configs/export/merge.yaml")
MERGED_DIR = Path("artifacts/merged/Quattro-Formaggi-12B-Logistics-v0.1")
VERIFY_FILE = "verify_merged.json"
GGUF_CONFIG = Path("configs/export/gguf.yaml")
GGUF_REPORT = "gguf_report.json"
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
    what.add_argument("--gguf", type=Path,
                      help="a GGUF file of `qf export gguf` (step 17): hash and metadata "
                      "against the HF model it was made from")  # fmt: skip
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
    if args.gguf is not None:
        return _verify_gguf(args.gguf, root)
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


# --- step 17: GGUF ------------------------------------------------------------------------------


def _gguf_config(path: Path, root: Path) -> ExportGgufConfig:
    return load_yaml_config(path if path.is_absolute() else root / path, ExportGgufConfig)


def _source_parents(source: Path, root: Path) -> list[str]:
    """The merged package (an `hf_model` artifact) or the pinned base directory (its weights
    were checked by `qf train fetch-base`): the parents of the GGUF files."""
    if (root / source / EXPORT_MANIFEST).is_file():
        ref = read_artifact(source, "hf_model", supported_versions("hf_model"), root=root)
        return [ref.sha256]
    base = load_yaml_config(root / BASE_MODEL_CONFIG, BaseModelConfig)
    provenance = root / source / "weights_provenance.json"
    if Path(source) != base.local_dir or not provenance.is_file():
        raise QFError(f"{source}: neither a merged package ({EXPORT_MANIFEST}) nor the pinned "
                      f"base with its weights ({base.local_dir})")  # fmt: skip
    return [sha256_file(provenance)]


def _gguf_facts(path: Path, hf_dir: Path, gguf_py: Path) -> dict[str, Any]:
    meta = read_gguf_metadata(path, gguf_py)
    return {"bytes": path.stat().st_size, "architecture": meta.get("general.architecture"),
            "file_type": meta.get("general.file_type"), "vocab_size": meta.get("vocab_size"),
            "pre_tokenizer": meta.get("tokenizer.ggml.pre"),
            "bos": meta.get("tokenizer.ggml.bos_token_id"),
            "eos": meta.get("tokenizer.ggml.eos_token_id"),
            "chat_template_sha256": meta.get("chat_template_sha256"),
            "problems": check_gguf_against_hf(meta, hf_dir)}  # fmt: skip


def configure_gguf(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True,
                        help="the merged package or the pinned base directory")  # fmt: skip
    parser.add_argument("--name", required=True,
                        help="file prefix: artifacts/gguf/<name>-<TYPE>.gguf")  # fmt: skip
    parser.add_argument("--qtypes", default="Q4_K_M",
                        help="comma-separated quantization types, e.g. Q4_K_M,Q5_K_M")  # fmt: skip
    parser.add_argument("--config", type=Path, default=GGUF_CONFIG,
                        help="the converter and the quantizer (pinned llama.cpp)")  # fmt: skip


def run_gguf(args: argparse.Namespace) -> int:
    """HF model -> GGUF (bf16) -> quantized GGUFs; metadata checked against the HF model."""
    root = project_root()
    cfg = _gguf_config(args.config, root)
    qtypes = [q.strip() for q in args.qtypes.split(",") if q.strip()]
    if not qtypes or any(not q.replace("_", "").isalnum() for q in qtypes):
        raise QFError(f"--qtypes {args.qtypes!r}: expected e.g. Q4_K_M,Q5_K_M")
    source = args.source if not args.source.is_absolute() else args.source.relative_to(root)
    parents = _source_parents(source, root)
    converter, quantizer = build(CONVERTERS, cfg.converter), build(QUANTIZERS, cfg.quantizer)
    run = start_run("export-gguf", root)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    full = converter.convert(source, cfg.out_dir / f"{args.name}-{cfg.outtype.upper()}.gguf",
                             outtype=cfg.outtype, run_id=run.run_id, parents=parents)  # fmt: skip
    refs = [full] + [quantizer.quantize(full, cfg.out_dir / f"{args.name}-{q}.gguf", qtype=q,
                                        run_id=run.run_id) for q in qtypes]  # fmt: skip
    report = {ref.path.as_posix(): {"sha256": ref.sha256,
                                    **_gguf_facts(root / ref.path, root / source,
                                                  root / cfg.gguf_py)}
              for ref in refs}  # fmt: skip
    atomic_write_text(run.run_dir / GGUF_REPORT, json.dumps(report, indent=2) + "\n")
    problems = {path: facts["problems"] for path, facts in report.items() if facts["problems"]}
    run.finish(
        status="failed" if problems else "completed",
        config={**cfg.model_dump(mode="json"), "source": source.as_posix(), "name": args.name,
                "qtypes": qtypes},
        data_hashes={source.as_posix(): parents[0], **{r.path.as_posix(): r.sha256 for r in refs}},
        metrics={path: {"bytes": f["bytes"], "problems": len(f["problems"])}
                 for path, f in report.items()},
    )  # fmt: skip
    for ref in refs:
        facts = report[ref.path.as_posix()]
        print(f"{ref.path}: {facts['bytes'] / 2**30:.1f} GiB, sha256 {ref.sha256[:12]}…, "
              f"{'metadata = HF' if not facts['problems'] else facts['problems']}")  # fmt: skip
    print(f"Report: {run.run_dir.relative_to(root) / GGUF_REPORT}")
    return 1 if problems else 0


def _verify_gguf(path: Path, root: Path) -> int:
    """A GGUF file (e.g. copied to the Mac): hash against its manifest, metadata against the
    HF model it names when that model is on this machine."""
    relative = path if not path.is_absolute() else path.relative_to(root)
    try:
        ref = read_artifact(relative, "gguf", supported_versions("gguf"), root=root)
    except QFError as exc:
        print(f"  PROBLEM: {exc}")
        print(f"{relative}: NOT intact")
        return 1
    print(f"  ok: sha256 {ref.sha256[:12]}… matches its manifest")
    extra = load_artifact_manifest(root / relative, root=root).extra
    hf_source = extra.get("hf_source")
    problems: list[str] = []
    if hf_source and (root / hf_source).is_dir():
        cfg = _gguf_config(GGUF_CONFIG, root)
        problems = check_gguf_against_hf(read_gguf_metadata(root / relative, root / cfg.gguf_py),
                                         root / hf_source)  # fmt: skip
        for line in problems:
            print(f"  PROBLEM: {line}")
        if not problems:
            print(f"  ok: chat template, BOS and EOS equal to {hf_source}")
    print(f"  llama.cpp {str(extra.get('llama_cpp_commit'))[:8]}, type {extra.get('type')}")
    print(f"{relative}: {'intact' if not problems else 'NOT intact'}")
    return 0 if not problems else 1
