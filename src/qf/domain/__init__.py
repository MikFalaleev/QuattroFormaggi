"""Domain layer: pure logistics logic (units, rules, geo, serialization, tasks).

Depends only on `qf.contracts` and `qf.common`; no I/O and no heavy libraries. Other packages
import from here, never from submodules.
"""

from qf.domain.geo import CITIES, EARTH_RADIUS_KM, CityInfo, canonical_place, great_circle_km
from qf.domain.prompting import (
    DEFAULT_PROMPT_VERSION,
    REQUEST_DATE_LABELS,
    build_messages,
    load_system_prompt,
    system_prompt_hash,
)
from qf.domain.record_checks import check_record
from qf.domain.records_jsonl import parse_sft_jsonl
from qf.domain.rules import (
    DEFAULT_PIECES,
    REQUIRED_ORDER,
    check_target_consistency,
    compute_missing_fields,
    effective_pieces,
    total_weight_kg,
)
from qf.domain.rules_v2 import (
    BASE_REQUIRED_ORDER_V2,
    CONDITION_MISSING_NAMES,
    REQUIRED_CONDITIONS,
    canonical_card_v2,
    check_target_consistency_v2,
    compute_missing_fields_v2,
    condition_lacks_values,
    has_special_conditions,
)
from qf.domain.schemas import (
    TARGET_SCHEMAS,
    TargetSchema,
    get_target_schema,
    register_target_schema,
)
from qf.domain.serialization import (
    TargetParseError,
    parse_target,
    parse_target_lenient,
    parse_target_lenient_v2,
    parse_target_v2,
    serialize_target,
)
from qf.domain.tasks import SHIPMENT_EXTRACTION, TASKS, TaskSpec, get_task, register_task
from qf.domain.units import (
    CM_TO_M,
    LB_TO_KG,
    T_TO_KG,
    format_kg,
    from_kg,
    normalize_number,
    render_value_from_lbs,
    render_value_per_piece,
    to_kg,
    to_m,
)

__all__ = [
    "BASE_REQUIRED_ORDER_V2",
    "CM_TO_M",
    "CONDITION_MISSING_NAMES",
    "REQUIRED_CONDITIONS",
    "TARGET_SCHEMAS",
    "TargetSchema",
    "canonical_card_v2",
    "check_target_consistency_v2",
    "compute_missing_fields_v2",
    "condition_lacks_values",
    "get_target_schema",
    "has_special_conditions",
    "parse_target_lenient_v2",
    "parse_target_v2",
    "register_target_schema",
    "to_m",
    "parse_sft_jsonl",
    "system_prompt_hash",
    "render_value_per_piece",
    "load_system_prompt",
    "great_circle_km",
    "build_messages",
    "REQUEST_DATE_LABELS",
    "EARTH_RADIUS_KM",
    "DEFAULT_PROMPT_VERSION",
    "CITIES",
    "DEFAULT_PIECES",
    "LB_TO_KG",
    "REQUIRED_ORDER",
    "SHIPMENT_EXTRACTION",
    "TASKS",
    "T_TO_KG",
    "CityInfo",
    "TargetParseError",
    "TaskSpec",
    "canonical_place",
    "check_record",
    "check_target_consistency",
    "compute_missing_fields",
    "effective_pieces",
    "format_kg",
    "from_kg",
    "get_task",
    "normalize_number",
    "parse_target",
    "parse_target_lenient",
    "register_task",
    "render_value_from_lbs",
    "serialize_target",
    "to_kg",
    "total_weight_kg",
]
