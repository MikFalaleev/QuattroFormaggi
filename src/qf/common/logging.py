"""Logging setup used by the CLI (debug output goes through logging, never print)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qf.common.errors import QFError

__all__ = ["LOG_FORMAT", "NOISY_LOGGERS", "setup_logging"]

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Third-party loggers that log every HTTP request at INFO; shown only with DEBUG.
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    level_name = level.upper()
    if level_name not in logging.getLevelNamesMapping():
        raise QFError(f"unknown log level {level!r}")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level_name, format=LOG_FORMAT, handlers=handlers, force=True)
    noisy_level = logging.DEBUG if level_name == "DEBUG" else logging.WARNING
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)
