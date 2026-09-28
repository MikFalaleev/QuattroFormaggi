"""Contract test of every registered merger (plan C.8, step 16): offline on a tiny random model
(skipped without torch). A merger returns an `hf_model` artifact in `out` and refuses an
adapter of another base model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import qf.cli.wiring  # noqa: F401  (every registry and lazy entry)
from qf.cli.wiring import build
from qf.common import ComponentConfig, QFError, read_artifact
from qf.contracts import Merger, ModelRef, supported_versions
from qf.export import MERGERS

OFFLINE_CONFIGS: dict[str, dict[str, Any]] = {"peft_merge": {}}  # filled from tests.tiny_hf
OUT = Path("artifacts/merged/contract")


def test_every_merger_has_an_offline_config() -> None:
    assert sorted(MERGERS.names()) == sorted(OFFLINE_CONFIGS)


@pytest.fixture
def tiny(fake_project: Path) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tests.tiny_hf import tiny_adapter, tiny_base

    tiny_base(fake_project)
    return tiny_adapter(fake_project)


def merger(name: str) -> Merger:
    from tests.tiny_hf import TINY_BASE, TINY_ID
    from tests.tiny_training import REVISION

    params = {"base_id": TINY_ID, "revision": REVISION, "base_dir": TINY_BASE.as_posix(),
              "dtype": "fp32", "device": "cpu", "allow_test_adapter": True,
              "tokenizer_files": ["tokenizer.json", "tokenizer_config.json"],
              **OFFLINE_CONFIGS[name]}  # fmt: skip
    instance = build(MERGERS, ComponentConfig(name=name, **params))
    assert isinstance(instance, Merger) and instance.name == name
    return instance


@pytest.mark.parametrize("name", MERGERS.names())
def test_merger_writes_an_hf_model_artifact(name: str, tiny: Path, fake_project: Path) -> None:
    from tests.tiny_hf import TINY_ID
    from tests.tiny_training import REVISION

    adapter = read_artifact(tiny, "lora_adapter", supported_versions("lora_adapter"))
    ref = merger(name).merge(ModelRef(id=TINY_ID, revision=REVISION), adapter, OUT, run_id="r")
    assert ref.kind == "hf_model" and ref.path.as_posix() == OUT.as_posix()
    assert read_artifact(OUT, "hf_model", supported_versions("hf_model")).sha256 == ref.sha256


@pytest.mark.parametrize("name", MERGERS.names())
def test_merger_refuses_another_base(name: str, tiny: Path) -> None:
    adapter = read_artifact(tiny, "lora_adapter", supported_versions("lora_adapter"))
    with pytest.raises(QFError):
        merger(name).merge(ModelRef(id="other/model", revision="d" * 40), adapter, OUT,
                           run_id="r")  # fmt: skip
