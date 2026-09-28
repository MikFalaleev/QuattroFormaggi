"""Step 16 (D-122) on a tiny random model on the CPU in fp32: the merged model answers as base +
adapter, the tokenizer files are the base's, the manifest lists every file, a merge that changes
nothing is refused or detected, and the commands `qf export merge|verify --merged`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from qf.backends.hf_local import HFLocalBackend, HFLocalConfig  # noqa: E402
from qf.cli.main import main  # noqa: E402
from qf.common import ArtifactRef, QFError, read_artifact, sha256_file, write_artifact  # noqa: E402
from qf.contracts import ModelRef, supported_versions  # noqa: E402
from qf.export import EXPORT_MANIFEST, verify_merged, verify_merged_files  # noqa: E402
from qf.export.mergers.peft_merge import PeftMergeConfig, PeftMerger  # noqa: E402
from tests.tiny_hf import (  # noqa: E402
    ADAPTER,
    TINY_BASE,
    TINY_ID,
    hf_config,
    tiny_adapter,
    tiny_base,
)
from tests.tiny_training import REVISION, SYSTEM, tiny_records, user_text  # noqa: E402

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
OUT = Path("artifacts/merged/tiny")
BASE = ModelRef(id=TINY_ID, revision=REVISION)
MESSAGES = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(i)}]
            for i in (1, 7, 12)]  # fmt: skip


def merger(**changes: Any) -> PeftMerger:
    params: dict[str, Any] = {"base_id": TINY_ID, "revision": REVISION,
                              "base_dir": TINY_BASE.as_posix(), "dtype": "fp32",
                              "device": "cpu", "tokenizer_files": TOKENIZER_FILES,
                              "allow_test_adapter": True}  # fmt: skip
    return PeftMerger(PeftMergeConfig(**(params | changes)))


def adapter_ref() -> Any:
    return read_artifact(ADAPTER, "lora_adapter", supported_versions("lora_adapter"))


@pytest.fixture
def root(fake_project: Path) -> Path:
    tiny_base(fake_project)
    return fake_project


def merged(root: Path, **changes: Any) -> ArtifactRef:
    tiny_adapter(root)
    return merger(**changes).merge(
        BASE, adapter_ref(), OUT, run_id="20260101-000000-export-merge-x"
    )


def test_merged_logits_equal_adapter_logits(root: Path) -> None:
    merged(root)
    ids = torch.tensor([[1, 8, 9, 10, 11, 12, 13, 3]])
    tuned = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(root / TINY_BASE),
                                      str(root / ADAPTER))  # fmt: skip
    full = AutoModelForCausalLM.from_pretrained(root / OUT)
    with torch.no_grad():
        expected, actual = tuned(input_ids=ids).logits, full(input_ids=ids).logits
        with tuned.disable_adapter():
            base = tuned(input_ids=ids).logits
    assert torch.allclose(actual, expected, atol=1e-4)
    assert not torch.allclose(actual, base, atol=1e-3)  # the merge changed the model


def test_tokenizer_unchanged_after_merge(root: Path) -> None:
    merged(root)
    for name in [*TOKENIZER_FILES, "generation_config.json"]:
        assert sha256_file(root / OUT / name) == sha256_file(root / TINY_BASE / name), name


def test_manifest_lists_all_files_with_hashes(root: Path) -> None:
    ref = merged(root)
    manifest = json.loads((root / OUT / EXPORT_MANIFEST).read_text(encoding="utf-8"))
    files = {p.relative_to(root / OUT).as_posix(): sha256_file(p)
             for p in (root / OUT).rglob("*")
             if p.is_file() and p.name != EXPORT_MANIFEST}  # fmt: skip
    assert manifest["files"] == files and "model.safetensors" in files
    assert manifest["adapter"]["sha256"] == adapter_ref().sha256
    assert set(manifest["package_versions"]) == {"torch", "transformers", "peft", "safetensors"}
    assert ref.kind == "hf_model" and ref.parents == (adapter_ref().sha256,)
    assert verify_merged_files(OUT, root, base_dir=root / TINY_BASE).ok


def test_changed_files_are_found(root: Path) -> None:
    merged(root)
    (root / OUT / "generation_config.json").write_text("{}", encoding="utf-8")
    check = verify_merged_files(OUT, root)
    assert not check.ok and "content hash mismatch" in check.problems[0]


def test_merge_refuses_an_adapter_with_zero_lora_b(root: Path) -> None:
    tiny_adapter(root, zero=True)
    with pytest.raises(QFError, match="all zero"):
        merger().merge(BASE, adapter_ref(), OUT, run_id="r")
    assert not (root / OUT).exists()


def test_merge_refuses_the_rehearsal_adapter_and_another_base(root: Path) -> None:
    tiny_adapter(root)
    with pytest.raises(QFError, match="loader 'tiny_random_cpu'"):
        merger(allow_test_adapter=False).merge(BASE, adapter_ref(), OUT, run_id="r")
    with pytest.raises(QFError, match="differs from the merger config"):
        merger().merge(ModelRef(id="other/model", revision="c" * 40), adapter_ref(), OUT,
                       run_id="r")  # fmt: skip


def test_merge_refuses_a_non_empty_output(root: Path) -> None:
    tiny_adapter(root)
    (root / OUT).mkdir(parents=True)
    (root / OUT / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(QFError, match="not empty"):
        merger().merge(BASE, adapter_ref(), OUT, run_id="r")


def test_verify_merged_passes(root: Path) -> None:
    merged(root)
    check = verify_merged(OUT, ADAPTER, TINY_BASE, root, messages=MESSAGES, max_new_tokens=8,
                          dtype="fp32", device="cpu")  # fmt: skip
    assert check.ok, check.problems
    assert all(p.greedy_equal for p in check.prompts)
    assert any(p.differs_from_base for p in check.prompts)
    assert max(p.max_abs_diff for p in check.prompts) < 1e-3


def test_verify_merged_catches_a_merge_that_changed_nothing(root: Path) -> None:
    """A "merged" package that is the base itself, with valid manifests."""
    tiny_adapter(root)
    shutil.copytree(root / TINY_BASE, root / OUT)
    files = {p.relative_to(root / OUT).as_posix(): sha256_file(p)
             for p in (root / OUT).rglob("*") if p.is_file()}  # fmt: skip
    own = {"files": files, "copied_from_base": [],
           "adapter": {"path": ADAPTER.as_posix(), "sha256": adapter_ref().sha256}}  # fmt: skip
    (root / OUT / EXPORT_MANIFEST).write_text(json.dumps(own), encoding="utf-8")
    write_artifact(root / OUT, "hf_model", "hf_safetensors_v1", "r", root=root)
    check = verify_merged(OUT, ADAPTER, TINY_BASE, root, messages=MESSAGES, max_new_tokens=8,
                          dtype="fp32", device="cpu")  # fmt: skip
    assert not check.ok
    assert any("answers as the base alone" in p for p in check.problems)
    diverged = [p for p in check.prompts if not p.greedy_equal]
    assert diverged and all(p.first_divergence is not None for p in diverged)
    assert all(p.margin_at_divergence is not None and p.margin_at_divergence >= 0
               for p in diverged)  # fmt: skip


def test_hf_local_names_a_merged_package(root: Path) -> None:
    ref = merged(root)
    impl = HFLocalBackend(HFLocalConfig(**hf_config(base_dir=OUT.as_posix())))
    assert impl.model_id() == f"merged {ref.sha256[:12]} of {TINY_ID}@{REVISION[:12]} fp32"


def project_configs(root: Path) -> None:
    (root / "configs/train").mkdir(parents=True, exist_ok=True)
    (root / "configs/train/base_model.yaml").write_text(
        f"id: {TINY_ID}\nrevision: {REVISION}\nlicense: apache-2.0\nfix_mistral_regex: false\n"
        f"tokenizer_files: [tokenizer.json]\nlocal_dir: {TINY_BASE.as_posix()}\n",
        encoding="utf-8")  # fmt: skip
    bench = root / "data/bench.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text("".join(r.model_dump_json() + "\n" for r in tiny_records(3, "test")),
                     encoding="utf-8")  # fmt: skip
    (root / "configs/export").mkdir(parents=True, exist_ok=True)
    (root / "configs/export/merge.yaml").write_text(
        f"merger:\n  name: peft_merge\n  base_id: {TINY_ID}\n  revision: {REVISION}\n"
        f"  base_dir: {TINY_BASE.as_posix()}\n  dtype: fp32\n  device: cpu\n"
        "  fix_mistral_regex: false\n"
        f"  tokenizer_files: [{', '.join(TOKENIZER_FILES)}]\n  allow_test_adapter: true\n"
        "verify:\n  bench: data/bench.jsonl\n  prompts: 3\n  max_new_tokens: 8\n",
        encoding="utf-8")  # fmt: skip


def test_cli_merge_and_verify(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tiny_adapter(root)
    project_configs(root)
    assert main(["export", "merge", "--adapter", ADAPTER.as_posix(), "--out", OUT.as_posix()]) == 0
    assert f"Merged: {OUT.as_posix()}" in capsys.readouterr().out
    assert main(["export", "verify", "--merged", OUT.as_posix(), "--files-only"]) == 0
    assert "intact (files)" in capsys.readouterr().out
    assert main(["export", "verify", "--merged", OUT.as_posix()]) == 0
    out = capsys.readouterr().out
    assert "verified" in out and "equal to base + adapter on 3 of 3 prompts" in out
    runs = sorted(p.name for p in (root / "runs").iterdir())
    assert any("export-merge" in r for r in runs) and any("export-verify" in r for r in runs)
