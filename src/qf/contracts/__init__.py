"""Contracts layer: data types and Protocol ports shared across packages (plan C.2).

Depends only on `qf.common`. Other packages import from here, never from submodules.
"""

from qf.contracts.generation import GenerationRequest, GenerationResult, Message, Role
from qf.contracts.versions import SUPPORTED_SCHEMA_VERSIONS, require_supported, supported_versions

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "GenerationRequest",
    "GenerationResult",
    "Message",
    "Role",
    "require_supported",
    "supported_versions",
]
