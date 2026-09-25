"""What a resume must keep (D-118): data, tokenizer, prompt, base revision, experiment,
critical library versions and — unless accepted with a reason — the code commit."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from qf.common import RunManifest  # noqa: E402
from qf.training.trainers.hf_qlora.train import _resume_problems  # noqa: E402

IDENTITY: dict[str, Any] = {
    "git_commit": "a" * 40, "git_dirty": False, "base_revision": "b" * 40,
    "data_hashes": {"data/train.jsonl": "1" * 64}, "tokenizer_hash": "t",
    "chat_template_hash": "c", "prompt_hashes": {"system": "p"}, "experiment_id": "E302",
    "package_versions": {"torch": "2.14.0", "transformers": "5.17.0", "peft": "0.21.0",
                         "accelerate": "1.15.0", "bitsandbytes": "0.50.2", "tokenizers": "0.23.2"},
}  # fmt: skip


def manifest(**changes: Any) -> RunManifest:
    fields = {key: value for key, value in IDENTITY.items() if key != "git_dirty"}
    return RunManifest(run_id="r", kind="train", created_at=datetime.now(UTC),
                       **{**fields, **changes})  # fmt: skip


def test_same_everything_resumes() -> None:
    assert _resume_problems(manifest(), IDENTITY, None) == []


def test_each_difference_is_reported() -> None:
    versions = {**IDENTITY["package_versions"], "transformers": "5.18.0", "tokenizers": "0.24"}
    previous = manifest(data_hashes={"data/train.jsonl": "2" * 64}, chat_template_hash="x",
                        experiment_id="E301", package_versions=versions)  # fmt: skip
    problems = " | ".join(_resume_problems(previous, IDENTITY, None))
    assert "data_hashes" in problems and "chat_template_hash" in problems
    assert "experiment_id: E301 → E302" in problems and "transformers 5.18.0 → 5.17.0" in problems
    assert "tokenizers" not in problems  # not critical for the optimizer and RNG states


def test_code_change_needs_a_reason() -> None:
    previous = manifest(git_commit="f" * 40)
    assert "code commit" in _resume_problems(previous, IDENTITY, None)[0]
    assert _resume_problems(previous, IDENTITY, "a fix of the logging only") == []
