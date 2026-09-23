from __future__ import annotations

from tests.architecture.helpers import lab_isolation_violations


def test_lab_isolated() -> None:
    """QF-Lab imports only qf.common; nothing imports QF-Lab (rule A.2.1)."""
    assert lab_isolation_violations() == []
