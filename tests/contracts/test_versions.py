from __future__ import annotations

import pytest

from qf.common import QFError
from qf.contracts import SUPPORTED_SCHEMA_VERSIONS, require_supported, supported_versions


def test_require_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SUPPORTED_SCHEMA_VERSIONS, "sft_dataset", frozenset({"sft_record_v1"}))
    require_supported("sft_dataset", "sft_record_v1")
    assert supported_versions("sft_dataset") == {"sft_record_v1"}
    with pytest.raises(QFError, match="unsupported schema version 'sft_record_v9'"):
        require_supported("sft_dataset", "sft_record_v9")


def test_unknown_kind_is_explicit_error() -> None:
    with pytest.raises(QFError, match="no supported schema versions"):
        require_supported("no_such_kind", "v1")
