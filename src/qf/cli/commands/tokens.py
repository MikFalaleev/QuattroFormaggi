"""`qf tokens …` (plan step 11): the pinned tokenizer files and the audit of the loss mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qf.common import (
    QFError,
    atomic_write_text,
    load_yaml_config,
    project_root,
    sha256_file,
    start_run,
)
from qf.domain import parse_sft_jsonl
from qf.training import (
    BaseModelConfig,
    audit_masks,
    build_features,
    chat_template_hash,
    choose_pad_token,
    download_tokenizer_files,
    load_tokenizer,
    render_audit,
    tokenizer_hash,
    verify_tokenizer_files,
)

__all__ = ["configure_audit", "configure_fetch", "run_audit", "run_fetch"]

DEFAULT_CONFIG = Path("configs/train/base_model.yaml")
DEFAULT_DATA = Path("data/processed/generated_v2")
SPLIT_FILES = ("train", "val", "test", "test_ood")


def configure_fetch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="base model config")


def run_fetch(args: argparse.Namespace) -> int:
    root = project_root()
    cfg = load_yaml_config(root / args.config, BaseModelConfig)
    directory = download_tokenizer_files(cfg, root)
    provenance = verify_tokenizer_files(cfg, root)
    for name, info in provenance["files"].items():
        print(f"  {name}: {info['bytes']} bytes, sha256 {info['sha256'][:12]}")
    print(f"Tokenizer files of {cfg.id}@{cfg.revision[:12]} (no weights): {directory}")
    return 0


def configure_audit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="base model config")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="dataset directory with train/val/test/test_ood .jsonl")  # fmt: skip
    parser.add_argument("--max-len", type=int, default=2048, help="max tokens per record")
    parser.add_argument("--examples", type=int, default=5, help="decoded examples in the report")


def _records(data: Path) -> tuple[list[Any], dict[str, str]]:
    records, hashes = [], {}
    for split in SPLIT_FILES:
        path = data / f"{split}.jsonl"
        if not path.exists():
            continue
        found, issues = parse_sft_jsonl(path.read_text(encoding="utf-8"), source=path.name,
                                        split=split)  # fmt: skip
        if issues:
            raise QFError(f"{path}: {len(issues)} invalid record(s), e.g. {issues[0]}")
        records += found
        hashes[path.as_posix()] = sha256_file(path)
    if not records:
        raise QFError(f"no records in {data} ({', '.join(SPLIT_FILES)})")
    return records, hashes


def run_audit(args: argparse.Namespace) -> int:
    root = project_root()
    cfg = load_yaml_config(root / args.config, BaseModelConfig)
    verify_tokenizer_files(cfg, root)
    directory = cfg.directory(root)
    tok = load_tokenizer(directory, fix_mistral_regex=cfg.fix_mistral_regex)
    data = args.data if args.data.is_absolute() else root / args.data
    records, hashes = _records(data)
    run = start_run("tokens-audit", root)
    report = audit_masks(records, tok, max_len=args.max_len, n_examples=args.examples)
    pad_id, pad_reason = choose_pad_token(tok)
    changed = _regex_fix_changes(records, directory, tok, args.max_len, cfg.fix_mistral_regex)
    hashes_meta = {"tokenizer_hash": tokenizer_hash(directory),
                   "chat_template_hash": chat_template_hash(tok)}  # fmt: skip
    meta = {"model": cfg.id, "revision": cfg.revision, **hashes_meta,
            "data": f"`{args.data.as_posix()}`"}  # fmt: skip
    summary = {**report.summary(), "pad_token_id": pad_id, "pad_token_reason": pad_reason,
               "vocab_size": len(tok), "records_changed_by_regex_fix": changed}  # fmt: skip
    run.run_dir.mkdir(parents=True, exist_ok=True)
    text = render_audit(report, meta) + (
        f"\n## Токенизатор\n\n- Pad: id {pad_id} — {pad_reason}; словарь не расширяется "
        f"({len(tok)} токенов).\n- Исправление правила разбиения (`fix_mistral_regex`) меняет "
        f"токены в {changed} записях из {report.records}.\n"
    )  # fmt: skip
    atomic_write_text(run.run_dir / "mask_audit.md", text)
    atomic_write_text(run.run_dir / "mask_audit.json",
                      json.dumps(summary, ensure_ascii=False, indent=2) + "\n")  # fmt: skip
    run.finish(config={"base_model": cfg.model_dump(mode="json"), "max_len": args.max_len},
               base_revision=cfg.revision, data_hashes=hashes, metrics=summary,
               packages=("transformers", "tokenizers"),
               tokenizer_hash=hashes_meta["tokenizer_hash"],
               chat_template_hash=hashes_meta["chat_template_hash"])  # fmt: skip
    print(f"{report.records} records: too long {summary['too_long']}, tokens p95 "
          f"{summary['tokens_p95']}, max {summary['tokens_max']}, trainable "
          f"{summary['trainable_share']:.1%}; template: {summary['template']}")  # fmt: skip
    print(f"Report: {run.run_dir / 'mask_audit.md'}")
    return 0


def _regex_fix_changes(records: list[Any], directory: Path, tok: Any, max_len: int,
                       fixed: bool) -> int:  # fmt: skip
    """Records whose ids differ between the tokenizer with and without `fix_mistral_regex`."""
    other = load_tokenizer(directory, fix_mistral_regex=not fixed)
    changed = 0
    for record in records:
        a, b = build_features(record, tok, max_len), build_features(record, other, max_len)
        changed += getattr(a, "input_ids", None) != getattr(b, "input_ids", None)
    return changed
