"""Contracts layer: data types and Protocol ports shared across packages (plan C.2).

Depends only on `qf.common`. Other packages import from here, never from submodules.
"""

from qf.contracts.card_v1 import SCHEMA_VERSION as CARD_SCHEMA_VERSION
from qf.contracts.card_v1 import (
    CargoCategory,
    Conflict,
    EquipmentType,
    ExtractionTarget,
    FieldName,
    Place,
    Quantity,
    ShipmentCard,
    WeightUnit,
)
from qf.contracts.facts import LOAD_FACTS_SCHEMA_VERSION, LoadFacts
from qf.contracts.generation import GenerationRequest, GenerationResult, Message, Role
from qf.contracts.ports import FactsBuilder, RawSource
from qf.contracts.records import (
    RECORD_ROLES,
    SFT_RECORD_SCHEMA_VERSION,
    HardCaseName,
    Language,
    SFTRecord,
    SplitName,
    TargetSchemaVersion,
    TaskName,
    VariantInfo,
)
from qf.contracts.versions import SUPPORTED_SCHEMA_VERSIONS, require_supported, supported_versions

__all__ = [
    "CARD_SCHEMA_VERSION",
    "LOAD_FACTS_SCHEMA_VERSION",
    "RECORD_ROLES",
    "SFT_RECORD_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "CargoCategory",
    "Conflict",
    "EquipmentType",
    "ExtractionTarget",
    "FactsBuilder",
    "FieldName",
    "GenerationRequest",
    "GenerationResult",
    "HardCaseName",
    "Language",
    "LoadFacts",
    "Message",
    "Place",
    "Quantity",
    "RawSource",
    "Role",
    "SFTRecord",
    "ShipmentCard",
    "SplitName",
    "TargetSchemaVersion",
    "TaskName",
    "VariantInfo",
    "WeightUnit",
    "require_supported",
    "supported_versions",
]
