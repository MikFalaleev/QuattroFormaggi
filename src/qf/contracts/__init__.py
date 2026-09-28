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
from qf.contracts.card_v2 import CONDITION_KINDS as CARD_V2_CONDITION_KINDS
from qf.contracts.card_v2 import SCHEMA_VERSION as CARD_V2_SCHEMA_VERSION
from qf.contracts.card_v2 import CargoCategory as CargoCategoryV2
from qf.contracts.card_v2 import (
    ConditionKind,
    Length,
    LengthUnit,
    OversizeCondition,
    PackagingCondition,
    PackagingType,
    SecuringCondition,
    SecuringMethod,
    SensorParameter,
    SensorsCondition,
    SpecialCondition,
    TemperatureCondition,
)
from qf.contracts.card_v2 import Conflict as ConflictV2
from qf.contracts.card_v2 import ConflictField as ConflictFieldV2
from qf.contracts.card_v2 import EquipmentType as EquipmentTypeV2
from qf.contracts.card_v2 import ExtractionTarget as ExtractionTargetV2
from qf.contracts.card_v2 import FieldName as FieldNameV2
from qf.contracts.card_v2 import MissingField as MissingFieldV2
from qf.contracts.card_v2 import ShipmentCard as ShipmentCardV2
from qf.contracts.evaluation import CaseScore, CaseScoreV2, MetricValue
from qf.contracts.facts import LOAD_FACTS_SCHEMA_VERSION, LoadFacts
from qf.contracts.facts_v2 import LOAD_FACTS_V2_SCHEMA_VERSION, AnyLoadFacts, LoadFactsV2
from qf.contracts.generation import GenerationRequest, GenerationResult, Message, Role
from qf.contracts.models import ModelRef
from qf.contracts.ports import (
    AdapterTrainer,
    BatchGenerationBackend,
    FactsBuilder,
    GenerationBackend,
    HardCase,
    Merger,
    Metric,
    RawSource,
    Splitter,
    TemplateFamily,
)
from qf.contracts.records import (
    RECORD_ROLES,
    SFT_RECORD_SCHEMA_VERSION,
    HardCaseName,
    Language,
    OodReason,
    SFTRecord,
    SplitName,
    TargetSchemaVersion,
    TaskName,
    VariantInfo,
)
from qf.contracts.rendering import (
    AnyFieldName,
    Dimension,
    RenderedField,
    RenderedRequest,
    RequestDraft,
)
from qf.contracts.training import (
    PLACEHOLDER_REVISION,
    DataHashes,
    Estimate,
    ExperimentSpec,
    MemoryEstimate,
    Preflight,
    TokenStats,
    TrainBudget,
    TrainConfig,
    TrainData,
    TrainLogging,
    TrainLora,
    TrainModel,
    TrainOutput,
    TrainQuantization,
    TrainSettings,
)
from qf.contracts.versions import SUPPORTED_SCHEMA_VERSIONS, require_supported, supported_versions

__all__ = [
    "PLACEHOLDER_REVISION",
    "AdapterTrainer",
    "DataHashes",
    "Estimate",
    "ExperimentSpec",
    "MemoryEstimate",
    "ModelRef",
    "Preflight",
    "TokenStats",
    "TrainBudget",
    "TrainConfig",
    "TrainData",
    "TrainLogging",
    "TrainLora",
    "TrainModel",
    "TrainOutput",
    "TrainQuantization",
    "TrainSettings",
    "MetricValue",
    "Metric",
    "BatchGenerationBackend",
    "GenerationBackend",
    "CaseScore",
    "CaseScoreV2",
    "TemplateFamily",
    "RequestDraft",
    "RenderedRequest",
    "RenderedField",
    "OodReason",
    "HardCase",
    "Merger",
    "CARD_SCHEMA_VERSION",
    "CARD_V2_CONDITION_KINDS",
    "CARD_V2_SCHEMA_VERSION",
    "CargoCategoryV2",
    "ConditionKind",
    "ConflictFieldV2",
    "ConflictV2",
    "EquipmentTypeV2",
    "ExtractionTargetV2",
    "FieldNameV2",
    "Length",
    "LengthUnit",
    "MissingFieldV2",
    "OversizeCondition",
    "PackagingCondition",
    "PackagingType",
    "SecuringCondition",
    "SecuringMethod",
    "SensorParameter",
    "SensorsCondition",
    "ShipmentCardV2",
    "SpecialCondition",
    "TemperatureCondition",
    "LOAD_FACTS_SCHEMA_VERSION",
    "LOAD_FACTS_V2_SCHEMA_VERSION",
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
    "LoadFactsV2",
    "AnyLoadFacts",
    "AnyFieldName",
    "Dimension",
    "Message",
    "Place",
    "Quantity",
    "RawSource",
    "Role",
    "SFTRecord",
    "ShipmentCard",
    "SplitName",
    "Splitter",
    "TargetSchemaVersion",
    "TaskName",
    "VariantInfo",
    "WeightUnit",
    "require_supported",
    "supported_versions",
]
