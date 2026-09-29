"""Contract test of every registered GGUF converter and quantizer (plan C.8, step 17), offline:
the external tool is replaced by a fake that writes a tiny GGUF. Each returns a `gguf`
artifact at `out` that names the pinned tool commit, and refuses another commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import qf.cli.wiring  # noqa: F401  (every registry and lazy entry)
from qf.cli.wiring import build
from qf.common import ComponentConfig, QFError, load_artifact_manifest
from qf.contracts import ModelConverter, Quantizer
from qf.export import CONVERTERS, QUANTIZERS
from qf.export.converters import llama_cpp
from tests.test_gguf_export import COMMIT, SOURCE, FakeTools, hf_dir

OFFLINE: dict[str, dict[str, Any]] = {"llama_cpp_convert": {}, "llama_cpp_quantize": {}}


def test_every_gguf_tool_has_an_offline_config() -> None:
    assert sorted([*CONVERTERS.names(), *QUANTIZERS.names()]) == sorted(OFFLINE)


@pytest.fixture
def fake_tools(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(llama_cpp, "head_commit", lambda directory: COMMIT)
    monkeypatch.setattr(llama_cpp.subprocess, "run", FakeTools())
    hf_dir(fake_project)
    return fake_project


def converter(name: str) -> ModelConverter:
    instance = build(CONVERTERS, ComponentConfig(name=name, commit=COMMIT, **OFFLINE[name]))
    assert isinstance(instance, ModelConverter) and instance.name == name
    return instance


def quantizer(name: str) -> Quantizer:
    instance = build(QUANTIZERS, ComponentConfig(name=name, commit=COMMIT, **OFFLINE[name]))
    assert isinstance(instance, Quantizer) and instance.name == name
    return instance


@pytest.mark.parametrize("name", CONVERTERS.names())
def test_converter_writes_a_gguf_artifact(name: str, fake_tools: Path) -> None:
    ref = converter(name).convert(SOURCE, Path("artifacts/gguf/c.gguf"), outtype="bf16",
                                  run_id="r", parents=[])  # fmt: skip
    assert ref.kind == "gguf" and ref.path.as_posix() == "artifacts/gguf/c.gguf"
    assert load_artifact_manifest(fake_tools / ref.path).extra["llama_cpp_commit"] == COMMIT


@pytest.mark.parametrize("name", QUANTIZERS.names())
def test_quantizer_writes_a_gguf_artifact(name: str, fake_tools: Path) -> None:
    full = converter("llama_cpp_convert")
    source = full.convert(SOURCE, Path("artifacts/gguf/c.gguf"), outtype="bf16", run_id="r",
                          parents=[])  # fmt: skip
    ref = quantizer(name).quantize(source, Path("artifacts/gguf/q.gguf"), qtype="Q4_K_M",
                                   run_id="r")  # fmt: skip
    assert ref.kind == "gguf" and ref.parents == (source.sha256,)


@pytest.mark.parametrize("name", CONVERTERS.names())
def test_converter_refuses_another_commit(name: str, fake_project: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setattr(llama_cpp, "head_commit", lambda directory: "f" * 40)
    with pytest.raises(QFError):
        converter(name).convert(SOURCE, Path("artifacts/gguf/c.gguf"), outtype="bf16",
                                run_id="r", parents=[])  # fmt: skip
