"""Schema versions the current code can read, per artifact kind (plan C.7).

This is the only place listing supported versions. Each step that introduces an artifact
format adds its entry; an unknown version is an explicit error, never a guess. The key
`target` is not an artifact kind: it lists the answer schemas inside records
(`SFTRecord.schema_version`, `TaskSpec.target_schema_version`).
"""

from __future__ import annotations

from qf.common import QFError

__all__ = ["SUPPORTED_SCHEMA_VERSIONS", "require_supported", "supported_versions"]

SUPPORTED_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {
    "raw_dataset": frozenset({"logistics_ops_csv_v1"}),  # step 2
    "metrics": frozenset(
        {
            "profile_report_v1",
            "validation_report_v1",
            "length_report_v1",
            "metrics_v1",
            "facts_v2_report_v1",
        }
    ),  # steps 3, 7, 9, sub-step V2
    "sft_dataset": frozenset({"sft_record_v1"}),  # step 4 (files are written from step 6)
    "target": frozenset({"card_v1", "card_v2"}),  # step 4, sub-step V1 (D-080)
    "load_facts": frozenset({"load_facts_v1", "load_facts_v2"}),  # step 5, sub-step V2
    "benchmark": frozenset({"sft_record_v1"}),  # step 7 reads it, step 8 writes it
    "predictions": frozenset({"predictions_v1"}),  # step 9
    "lora_adapter": frozenset({"peft_lora_v1"}),  # step 12
    "hf_model": frozenset({"hf_safetensors_v1"}),  # step 16: merged weights
    "gguf": frozenset({"gguf_v3"}),  # step 17: llama.cpp GGUF files
}


def supported_versions(kind: str) -> frozenset[str]:
    versions = SUPPORTED_SCHEMA_VERSIONS.get(kind)
    if not versions:
        raise QFError(f"no supported schema versions are registered for artifact kind '{kind}'")
    return versions


def require_supported(kind: str, version: str) -> None:
    versions = supported_versions(kind)
    if version not in versions:
        listed = ", ".join(sorted(versions))
        raise QFError(f"unsupported schema version '{version}' for '{kind}' (supported: {listed})")
