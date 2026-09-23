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
    "metrics": frozenset({"profile_report_v1"}),  # step 3
    "sft_dataset": frozenset({"sft_record_v1"}),  # step 4 (files are written from step 6)
    "target": frozenset({"card_v1"}),  # step 4
    "load_facts": frozenset({"load_facts_v1"}),  # step 5
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
