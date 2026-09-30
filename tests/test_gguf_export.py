"""Step 17 (D-123) without llama.cpp tools: the commands built for the converter and the
quantizer, failures with the stderr tail, the pinned commit, the metadata check against the HF
tokenizer, reading a tiny GGUF, and `qf export gguf` / `qf export verify --gguf` end to end
with the tools replaced by fakes that write tiny GGUF files."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qf.cli.main import main
from qf.common import QFError, read_artifact, write_artifact
from qf.contracts import supported_versions
from qf.export import EXPORT_MANIFEST, check_gguf_against_hf, hf_tokenizer_facts, read_gguf_metadata
from qf.export.converters import llama_cpp
from qf.export.converters.llama_cpp import (
    LlamaCppConvertConfig,
    LlamaCppConverter,
    LlamaCppQuantizeConfig,
    LlamaCppQuantizer,
)
from tests.conftest import REPO_ROOT

GGUF_PY = REPO_ROOT / "third_party/llama.cpp/gguf-py"
COMMIT = "a" * 40
SOURCE = Path("artifacts/merged/tiny")
TEMPLATE = "{{ bos_token }}{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"


def hf_dir(root: Path) -> Path:
    """A tiny merged package: tokenizer files (BOS <s> = 1, EOS </s> = 2) and its manifests."""
    directory = root / SOURCE
    directory.mkdir(parents=True)
    config = {"bos_token": "<s>", "eos_token": "</s>", "chat_template": TEMPLATE}
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    added = [{"id": i, "content": c} for i, c in enumerate(["<unk>", "<s>", "</s>"])]
    (directory / "tokenizer.json").write_text(json.dumps({"added_tokens": added}),
                                              encoding="utf-8")  # fmt: skip
    (directory / EXPORT_MANIFEST).write_text("{}", encoding="utf-8")
    write_artifact(directory, "hf_model", "hf_safetensors_v1", "r-merge", root=root)
    return directory


def tiny_gguf(path: Path, *, template: str = TEMPLATE, eos: int = 2) -> None:
    if not GGUF_PY.is_dir():
        pytest.skip("third_party/llama.cpp is not checked out")
    sys.path.insert(0, str(GGUF_PY))
    import gguf

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_chat_template(template)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(eos)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["<unk>", "<s>", "</s>"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llama_cpp, "head_commit", lambda directory: COMMIT)


class FakeTools:
    """`subprocess.run` of the converter/quantizer: records calls and writes the output."""

    def __init__(self, fail: str | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.fail = fail

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        if self.fail:
            raise subprocess.CalledProcessError(1, args, output="", stderr=self.fail)
        if args[1].endswith("convert_hf_to_gguf.py"):
            tiny_gguf(Path(args[args.index("--outfile") + 1]))
        else:
            shutil.copyfile(args[1], args[2])
        return subprocess.CompletedProcess(args, 0, stdout="done", stderr="")


def converter() -> LlamaCppConverter:
    return LlamaCppConverter(LlamaCppConvertConfig(commit=COMMIT))


def quantizer() -> LlamaCppQuantizer:
    return LlamaCppQuantizer(LlamaCppQuantizeConfig(commit=COMMIT))


@pytest.mark.usefixtures("pinned")
def test_gguf_commands_built_correctly(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf_dir(fake_project)
    tools = FakeTools()
    monkeypatch.setattr(llama_cpp.subprocess, "run", tools)
    full = converter().convert(SOURCE, Path("artifacts/gguf/t-BF16.gguf"), outtype="bf16",
                               run_id="r1", parents=["p" * 64])  # fmt: skip
    quantizer().quantize(full, Path("artifacts/gguf/t-Q4_K_M.gguf"), qtype="Q4_K_M", run_id="r1")
    (convert, c_kwargs), (quantize, q_kwargs) = tools.calls
    root = fake_project
    assert convert == [str(root / ".venv-convert/bin/python"),
                       str(root / "third_party/llama.cpp/convert_hf_to_gguf.py"),
                       str(root / SOURCE), "--outtype", "bf16",
                       "--outfile", str(root / "artifacts/gguf/t-BF16.gguf")]  # fmt: skip
    assert quantize == [str(root / "third_party/llama.cpp-build/bin/llama-quantize"),
                        str(root / "artifacts/gguf/t-BF16.gguf"),
                        str(root / "artifacts/gguf/t-Q4_K_M.gguf"), "Q4_K_M"]  # fmt: skip
    assert c_kwargs["check"] is True and q_kwargs["check"] is True
    assert c_kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    manifest = json.loads((root / "artifacts/gguf/t-Q4_K_M.gguf.manifest.json").read_text())
    assert manifest["ref"]["kind"] == "gguf" and manifest["ref"]["parents"] == [full.sha256]
    assert manifest["extra"]["llama_cpp_commit"] == COMMIT
    assert manifest["extra"]["hf_source"] == SOURCE.as_posix()
    assert (root / "runs/r1/convert-t-BF16.gguf.log").read_text().startswith("$ ")


@pytest.mark.usefixtures("pinned")
def test_convert_failure_propagates(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf_dir(fake_project)
    monkeypatch.setattr(
        llama_cpp.subprocess, "run", FakeTools(fail="lots\nof\nlog\nKeyError: 'rope'")
    )
    with pytest.raises(QFError, match=r"exit code 1:[\s\S]*KeyError: 'rope'"):
        converter().convert(SOURCE, Path("artifacts/gguf/t-BF16.gguf"), outtype="bf16",
                            run_id="r1", parents=[])  # fmt: skip
    assert "KeyError" in (fake_project / "runs/r1/convert-t-BF16.gguf.log").read_text()


def test_another_llama_cpp_commit_is_refused(fake_project: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setattr(llama_cpp, "head_commit", lambda directory: "b" * 40)
    with pytest.raises(QFError, match="the config pins"):
        converter().convert(SOURCE, Path("artifacts/gguf/t.gguf"), outtype="bf16", run_id="r",
                            parents=[])  # fmt: skip


@pytest.mark.usefixtures("pinned")
def test_existing_output_is_refused(fake_project: Path) -> None:
    (fake_project / "artifacts/gguf").mkdir(parents=True)
    (fake_project / "artifacts/gguf/t.gguf").write_bytes(b"x")
    with pytest.raises(QFError, match="already exists"):
        converter().convert(SOURCE, Path("artifacts/gguf/t.gguf"), outtype="bf16", run_id="r",
                            parents=[])  # fmt: skip


def test_check_gguf_metadata_mismatch_detected(fake_project: Path) -> None:
    directory = hf_dir(fake_project)
    facts = hf_tokenizer_facts(directory)
    good = {"chat_template_sha256": facts["chat_template_sha256"],
            "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2}  # fmt: skip
    assert check_gguf_against_hf(good, directory) == []
    bad = good | {"chat_template_sha256": "0" * 64, "tokenizer.ggml.eos_token_id": 7}
    problems = check_gguf_against_hf(bad, directory)
    assert len(problems) == 2 and "chat template" in problems[0] and "eos_token_id 7" in problems[1]


def test_read_gguf_metadata_of_a_tiny_file(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    tiny_gguf(path)
    meta = read_gguf_metadata(path, GGUF_PY)
    assert meta["general.architecture"] == "llama" and meta["vocab_size"] == 3
    assert meta["tokenizer.ggml.bos_token_id"] == 1 and meta["tokenizer.ggml.eos_token_id"] == 2
    assert meta["tokenizer.chat_template"] == TEMPLATE


def gguf_config(root: Path) -> None:
    path = root / "configs/export/gguf.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    common = f"llama_cpp_dir: third_party/llama.cpp\n  commit: {COMMIT}\n"
    path.write_text(f"converter:\n  name: llama_cpp_convert\n  {common}"
                    f"quantizer:\n  name: llama_cpp_quantize\n  {common}"
                    f"gguf_py: {GGUF_PY}\n", encoding="utf-8")  # fmt: skip


@pytest.mark.usefixtures("pinned")
def test_cli_gguf_and_verify(fake_project: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    hf_dir(fake_project)
    gguf_config(fake_project)
    monkeypatch.setattr(llama_cpp.subprocess, "run", FakeTools())
    argv = ["export", "gguf", "--source", SOURCE.as_posix(), "--name", "t", "--qtypes", "Q4_K_M"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "artifacts/gguf/t-BF16.gguf" in out and "artifacts/gguf/t-Q4_K_M.gguf" in out
    assert out.count("metadata = HF") == 2
    ref = read_artifact(Path("artifacts/gguf/t-Q4_K_M.gguf"), "gguf", supported_versions("gguf"))
    assert main(["export", "verify", "--gguf", ref.path.as_posix()]) == 0
    assert "chat template, BOS and EOS equal to" in capsys.readouterr().out
    (fake_project / ref.path).write_bytes(b"broken")
    assert main(["export", "verify", "--gguf", ref.path.as_posix()]) == 1


@pytest.mark.usefixtures("pinned")
def test_cli_gguf_qtypes_none_converts_only(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-129: restores the BF16 parent of a quantized file where no llama-quantize exists."""
    hf_dir(fake_project)
    gguf_config(fake_project)
    calls = FakeTools()
    monkeypatch.setattr(llama_cpp.subprocess, "run", calls)
    argv = ["export", "gguf", "--source", SOURCE.as_posix(), "--name", "t", "--qtypes", "none"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "artifacts/gguf/t-BF16.gguf" in out and "Q4_K_M" not in out
    assert not any("quantize" in " ".join(map(str, args)) for args, _ in calls.calls)
    assert any(args[1].endswith("convert_hf_to_gguf.py") for args, _ in calls.calls)
    assert (
        main(["export", "gguf", "--source", SOURCE.as_posix(), "--name", "u", "--qtypes", ""]) == 1
    )  # an empty list is still refused


@pytest.mark.usefixtures("pinned")
def test_cli_gguf_fails_on_a_metadata_mismatch(
    fake_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_dir(fake_project)
    gguf_config(fake_project)

    class WrongEos(FakeTools):
        def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if args[1].endswith("convert_hf_to_gguf.py"):
                tiny_gguf(Path(args[args.index("--outfile") + 1]), eos=5)
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return super().__call__(args, **kwargs)

    monkeypatch.setattr(llama_cpp.subprocess, "run", WrongEos())
    assert main(["export", "gguf", "--source", SOURCE.as_posix(), "--name", "t"]) == 1
    runs = [p for p in (fake_project / "runs").iterdir() if "export-gguf" in p.name]
    assert json.loads((runs[0] / "run_manifest.json").read_text())["status"] == "failed"


def test_gguf_source_must_be_merged_or_the_pinned_base(fake_project: Path) -> None:
    (fake_project / "configs/train").mkdir(parents=True)
    (fake_project / "configs/train/base_model.yaml").write_text(
        "id: org/model\nrevision: " + "c" * 40 + "\nlicense: x\ntokenizer_files: [a]\n"
        "local_dir: artifacts/base_model/x\n", encoding="utf-8")  # fmt: skip
    gguf_config(fake_project)
    (fake_project / "somewhere").mkdir()
    assert main(["export", "gguf", "--source", "somewhere", "--name", "t"]) == 1


def test_tool_output_that_is_not_utf8_does_not_fail(tmp_path: Path) -> None:
    """llama-quantize prints raw token bytes: a real subprocess with invalid UTF-8 output."""
    noisy = (
        "import sys; sys.stdout.buffer.write(b'tensor \\xc4 ok'); sys.stderr.buffer.write(b'\\xff')"
    )
    llama_cpp.run_tool([sys.executable, "-c", noisy], tmp_path / "tool.log")
    assert "tensor" in (tmp_path / "tool.log").read_text(encoding="utf-8")
