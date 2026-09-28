"""`configs/export/merge.yaml` (plan step 16, D-122): the merger and the check of its result."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from qf.common import ComponentConfig, StrictConfig

__all__ = ["EXPORT_MANIFEST", "ExportMergeConfig", "MergeTarget", "VerifySettings"]

EXPORT_MANIFEST = "qf_export_manifest.json"
"""Written into the merged directory: base, adapter, dtype, sha256 of every file, versions."""


class MergeTarget(StrictConfig):
    """What every merger needs from its parameters: the base's local files, dtype, device."""

    base_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_dir: Path  # tokenizer, config and HF weights, relative to the project root
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    device: Literal["cuda", "cpu"] = "cuda"
    fix_mistral_regex: bool = True  # the tokenizer as training loads it (BaseModelConfig, D-113)

    @classmethod
    def of(cls, selector: ComponentConfig) -> MergeTarget:
        params: dict[str, Any] = selector.model_extra or {}
        return cls.model_validate({k: v for k, v in params.items() if k in cls.model_fields})


class VerifySettings(StrictConfig):
    """`qf export verify --merged` (D-122): the first `prompts` records of `bench` (never the
    gold answer), greedy `max_new_tokens` from base + adapter, from the merged model, and from
    the base alone (the adapter disabled), compared."""

    bench: Path = Path("data/splits_v2/bench_v2.jsonl")
    prompts: int = Field(default=5, ge=1)
    max_new_tokens: int = Field(default=32, ge=1)


class ExportMergeConfig(StrictConfig):
    merger: ComponentConfig
    verify: VerifySettings = Field(default_factory=VerifySettings)
