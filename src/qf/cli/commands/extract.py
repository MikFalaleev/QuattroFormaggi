"""`qf extract`: a request text -> a checked shipment card (step 18)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from qf.backends import BACKENDS
from qf.cli.wiring import build
from qf.common import QFError, load_yaml_config, project_root
from qf.runtime import ExtractConfig, ExtractionResult, detect_language, extract_card

__all__ = ["configure", "run"]

DEFAULT_CONFIG = Path("configs/runtime/extract.yaml")


def configure(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="the request text")
    source.add_argument("--file", type=Path, help="a file with the request text (UTF-8)")
    parser.add_argument("--request-date", type=date.fromisoformat, default=None,
                        metavar="YYYY-MM-DD",
                        help="date of the request (default: today)")  # fmt: skip
    parser.add_argument("--language", choices=("auto", "ru", "en"), default="auto",
                        help="language of the request (default: by its letters)")  # fmt: skip
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"runtime config (default: {DEFAULT_CONFIG})")  # fmt: skip
    parser.add_argument("--json", action="store_true", help="print the result as JSON")


def run(args: argparse.Namespace) -> int:
    root = project_root()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    if not text.strip():
        raise QFError("the request text is empty")
    cfg = load_yaml_config(args.config if args.config.is_absolute() else root / args.config,
                           ExtractConfig)  # fmt: skip
    language = detect_language(text) if args.language == "auto" else args.language
    result = extract_card(
        text, args.request_date or date.today(), language, build(BACKENDS, cfg.backend),
        settings=cfg.settings, schema_version=cfg.schema_version, task=cfg.task,
    )  # fmt: skip
    print(result.model_dump_json(indent=2) if args.json else _format(result))
    return 0 if result.status == "ok" else 1


def _format(result: ExtractionResult) -> str:
    lines = [f"Статус: {result.status}  (модель {result.model_id}, схема {result.schema_version})"]
    if result.card is not None:
        lines.append(json.dumps(result.card, ensure_ascii=False, indent=2))
        lines.append(f"Не хватает: {', '.join(result.missing_fields) or '—'}")
        derived = result.derived
        lines.append(f"Вес, кг: {derived.weight_total_kg}; мест: {derived.pieces}")
    if result.conflicts:
        lines.append(f"Противоречия: {json.dumps(result.conflicts, ensure_ascii=False)}")
    for item in result.assumptions:
        lines.append(f"Допущение: {item}")
    for item in result.warnings:
        lines.append(f"Предупреждение: {item}")
    if result.error:
        lines.append(f"Ошибка: {result.error}")
    lines.append(f"Нужна проверка человеком: {'да' if result.review_required else 'нет'}")
    return "\n".join(lines)
