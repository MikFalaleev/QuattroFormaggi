from __future__ import annotations

from pathlib import Path

import pytest

from tests.architecture.helpers import PACKAGES, SRC, imports_of, public_api_violations


@pytest.mark.parametrize("package", PACKAGES)
def test_cross_package_imports_use_public_api(package: str) -> None:
    assert public_api_violations(package) == []


def test_every_package_declares_public_api() -> None:
    for package in PACKAGES:
        source = (SRC / package / "__init__.py").read_text(encoding="utf-8")
        assert "__all__" in source, f"qf.{package} has no __all__"


def test_checker_detects_deep_and_submodule_imports(tmp_path: Path) -> None:
    src = tmp_path / "qf"
    for package in ("common", "domain"):
        (src / package).mkdir(parents=True)
        (src / package / "__init__.py").write_text("__all__ = []\n")
    (src / "common" / "errors.py").write_text("class QFError(Exception): ...\n")
    probe = src / "domain" / "probe.py"
    probe.write_text(
        "from qf.common.errors import QFError\n"
        "from qf.common import errors\n"
        "from qf.common import QFError as Ok\n"
        "from . import probe_sibling\n",
        encoding="utf-8",
    )
    refs = imports_of(probe, src)
    assert [r.module for r in refs] == ["qf.common.errors", "qf.common", "qf.common", "qf.domain"]
    violations = public_api_violations("domain", src)
    assert len(violations) == 2
    assert "qf.common.errors" in violations[0]
    assert "qf.common (errors)" in violations[1]
