"""`peft_merge`: the LoRA adapter merged into the full-precision base (plan step 16, D-122).

The base is loaded without quantization (bf16 by default) straight onto the device, the adapter
with `PeftModel`, then `merge_and_unload(safe_merge=True)` and `save_pretrained` in safetensors
shards. A merge that silently changes nothing is refused: an adapter that fails
`verify_adapter` (another base, the CPU rehearsal's loader), all-zero `lora_B` weights, LoRA
keys left in the result, a key set other than the base's, an unchanged `q_proj` weight. The
base's tokenizer files and `generation_config.json` are copied byte for byte after saving
(`save_pretrained` of transformers 5 may rewrite `tokenizer_config.json`), then every file is
hashed into `qf_export_manifest.json`. Registered lazily in `cli.wiring`.
"""

from __future__ import annotations

import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import torch
from pydantic import Field

from qf.common import ArtifactRef, QFError, project_root, sha256_file, write_artifact
from qf.contracts import ModelRef
from qf.export.adapters import verify_adapter
from qf.export.config import EXPORT_MANIFEST, MergeTarget

__all__ = ["DTYPES", "PeftMergeConfig", "PeftMerger", "load_causal_lm"]

DTYPES: Final[dict[str, Any]] = {"bf16": torch.bfloat16, "fp16": torch.float16,
                                 "fp32": torch.float32}  # fmt: skip
PACKAGES: Final = ("torch", "transformers", "peft", "safetensors")


class PeftMergeConfig(MergeTarget):
    # copied byte for byte from base_dir (with generation_config.json when the base has one)
    tokenizer_files: list[str] = Field(
        default_factory=lambda: ["tokenizer.json", "tokenizer_config.json",
                                 "special_tokens_map.json"], min_length=1)  # fmt: skip
    max_shard_size: str = "5GB"
    allow_test_adapter: bool = False  # tests only: an adapter of the CPU rehearsal


def load_causal_lm(directory: Path, dtype: str, device: str) -> Any:
    """A full-precision HF model from local files, straight onto the device."""
    from transformers import AutoModelForCausalLM

    if device == "cuda" and not torch.cuda.is_available():
        raise QFError("device cuda requested, but CUDA is not available")
    placement = {"device_map": {"": 0}} if device == "cuda" else {}
    return AutoModelForCausalLM.from_pretrained(str(directory), dtype=DTYPES[dtype],
                                                local_files_only=True, **placement)  # fmt: skip


def _probe_name(state: dict[str, Any]) -> str:
    names = [n for n in state if n.endswith("q_proj.weight")]
    if not names:
        raise QFError("the base has no q_proj weight to check the merge against")
    return sorted(names)[0]


class PeftMerger:
    """`Merger` over PEFT: base (full precision) + LoRA -> merged HF safetensors."""

    Config = PeftMergeConfig

    def __init__(self, cfg: PeftMergeConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "peft_merge"

    def _check_adapter(self, base: ModelRef, adapter: ArtifactRef, root: Path) -> None:
        check = verify_adapter(Path(adapter.path), root, base=base)
        problems = [p for p in check.problems
                    if not (self.cfg.allow_test_adapter and p.startswith("loader "))]  # fmt: skip
        if problems:
            raise QFError(f"{adapter.path}: not merged: {'; '.join(problems)}")
        if check.sha256 != adapter.sha256:
            raise QFError(f"{adapter.path}: sha256 {check.sha256} differs from the reference")

    def merge(self, base: ModelRef, adapter: ArtifactRef, out: Path, *,
              run_id: str) -> ArtifactRef:  # fmt: skip
        os.environ["HF_HUB_OFFLINE"] = "1"  # before peft/transformers import the Hub client
        from peft import PeftModel

        root = project_root()
        if (base.id, base.revision) != (self.cfg.base_id, self.cfg.revision):
            raise QFError(f"base {base.id}@{base.revision} differs from the merger config")
        self._check_adapter(base, adapter, root)
        base_dir, out_dir = root / self.cfg.base_dir, root / out
        if out_dir.exists() and any(out_dir.iterdir()):
            raise QFError(f"{out} is not empty: choose another directory")
        missing = [f for f in self.cfg.tokenizer_files if not (base_dir / f).is_file()]
        if missing:
            raise QFError(f"{self.cfg.base_dir}: missing tokenizer files {missing}")

        model = load_causal_lm(base_dir, self.cfg.dtype, self.cfg.device)
        base_keys = set(model.state_dict())
        probe = _probe_name(model.state_dict())
        before = model.state_dict()[probe].detach().to("cpu", copy=True)
        tuned = PeftModel.from_pretrained(model, str(root / adapter.path), is_trainable=False)
        lora_b = [p for n, p in tuned.named_parameters() if "lora_B" in n]
        if not lora_b or all(not bool(p.detach().abs().sum() > 0) for p in lora_b):
            raise QFError(f"{adapter.path}: the lora_B weights are all zero — nothing to merge")
        merged = tuned.merge_and_unload(safe_merge=True)
        state = merged.state_dict()
        leftover = [n for n in state if "lora_" in n]
        if leftover or set(state) != base_keys:
            raise QFError(f"merge left LoRA keys {leftover[:3]} or changed the key set "
                          f"({len(set(state) ^ base_keys)} differ)")  # fmt: skip
        if torch.equal(state[probe].detach().to("cpu"), before):
            raise QFError(f"merge changed nothing: {probe} is equal to the base")

        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_dir), safe_serialization=True,
                               max_shard_size=self.cfg.max_shard_size)  # fmt: skip
        copied = [*self.cfg.tokenizer_files]
        if (base_dir / "generation_config.json").is_file():
            copied.append("generation_config.json")
        for name in copied:
            shutil.copyfile(base_dir / name, out_dir / name)
        files = {p.relative_to(out_dir).as_posix(): sha256_file(p)
                 for p in sorted(out_dir.rglob("*")) if p.is_file()}  # fmt: skip
        manifest = {
            "base_model": base.id, "revision": base.revision,
            "adapter": {"path": adapter.path.as_posix(), "sha256": adapter.sha256,
                        "run_id": adapter.producer_run_id},
            "merger": self.name, "dtype": self.cfg.dtype, "safe_merge": True,
            "copied_from_base": copied, "files": files,
            "package_versions": {name: version(name) for name in PACKAGES},
        }  # fmt: skip
        (out_dir / EXPORT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")  # fmt: skip
        return write_artifact(out_dir, "hf_model", "hf_safetensors_v1", run_id, [adapter.sha256],
                              extra={"base_model": base.id, "revision": base.revision,
                                     "adapter_sha256": adapter.sha256, "dtype": self.cfg.dtype},
                              root=root)  # fmt: skip
