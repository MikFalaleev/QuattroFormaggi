from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from qf.common import QFError, setup_logging
from qf.common.logging import NOISY_LOGGERS


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_noisy = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    for name, level in saved_noisy.items():
        logging.getLogger(name).setLevel(level)
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


@pytest.mark.usefixtures("restore_root_logger")
def test_setup_logging_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    setup_logging("debug", log_file)
    logging.getLogger("qf.test").debug("hello from qf")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from qf" in log_file.read_text(encoding="utf-8")


@pytest.mark.usefixtures("restore_root_logger")
def test_http_libraries_are_quiet_unless_debug() -> None:
    setup_logging("INFO")
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    setup_logging("DEBUG")
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.DEBUG


def test_invalid_level_rejected() -> None:
    with pytest.raises(QFError, match="unknown log level"):
        setup_logging("LOUD")
