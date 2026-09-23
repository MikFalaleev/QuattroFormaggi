from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from qf.common import QFError, setup_logging


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield
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


def test_invalid_level_rejected() -> None:
    with pytest.raises(QFError, match="unknown log level"):
        setup_logging("LOUD")
