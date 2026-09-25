"""Integrity of a saved adapter without torch (D-118): the safetensors reader and the checks."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from qf.common import write_artifact
from qf.training import read_safetensors, verify_adapter


def safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, body = {}, b""
    for name, (dtype, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [len(body), len(body) + len(raw)]}  # fmt: skip
        body += raw
    encoded = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def bf16(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")


def test_read_safetensors_f32_and_bf16(tmp_path: Path) -> None:
    a = np.arange(6, dtype="<f4").reshape(2, 3)
    safetensors(tmp_path / "t.safetensors", {"a": ("F32", a), "b": ("BF16", bf16(a))})
    read = read_safetensors(tmp_path / "t.safetensors")
    assert np.array_equal(read["a"], a) and np.array_equal(read["b"], a)


def adapter(root: Path, tensors: dict[str, np.ndarray], base: str = "org/model",
            revision: str | None = "r" * 40) -> Path:  # fmt: skip
    directory = root / "runs" / "r" / "adapter"
    directory.mkdir(parents=True)
    safetensors(directory / "adapter_model.safetensors",
                {name: ("F32", value) for name, value in tensors.items()})  # fmt: skip
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "revision": revision})
    )
    write_artifact(directory, "lora_adapter", "peft_lora_v1", "r", root=root)
    return directory


LORA = {"x.lora_A.weight": np.ones((2, 4), "<f4"), "x.lora_B.weight": np.zeros((4, 2), "<f4")}


def test_a_good_adapter_passes(fake_project: Path) -> None:
    check = verify_adapter(adapter(fake_project, LORA), fake_project, expected_parameters=16,
                           base_model="org/model", revision="r" * 40)  # fmt: skip
    assert check.ok and check.parameters == 16 and check.tensors == 2


def test_problems_are_listed(fake_project: Path) -> None:
    bad = {**LORA, "x.lora_B.weight": np.full((4, 2), np.nan, "<f4"), "head.weight": np.ones(3)}
    directory = adapter(fake_project, {k: v.astype("<f4") for k, v in bad.items()})
    check = verify_adapter(directory, fake_project, expected_parameters=16,
                           base_model="other/model", revision="r" * 40)  # fmt: skip
    assert not check.ok
    text = " ".join(check.problems)
    assert "names org/model" in text and "outside LoRA" in text
    assert "19 parameters, expected 16" in text and "NaN or inf" in text


def test_changed_file_fails_the_hash(fake_project: Path) -> None:
    directory = adapter(fake_project, LORA)
    (directory / "adapter_config.json").write_text("{}")
    check = verify_adapter(directory, fake_project, expected_parameters=None,
                           base_model="org/model", revision="r" * 40)  # fmt: skip
    assert not check.ok and "content hash mismatch" in check.problems[0]
