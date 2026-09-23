"""Domain layer: pure logistics logic (units, rules, geo, serialization, tasks).

Depends only on `qf.contracts` and `qf.common`; no I/O and no heavy libraries. Other packages
import from here, never from submodules.
"""

from qf.domain.geo import CITIES, CityInfo, canonical_place
from qf.domain.record_checks import check_record
from qf.domain.rules import (
    DEFAULT_PIECES,
    REQUIRED_ORDER,
    check_target_consistency,
    compute_missing_fields,
    effective_pieces,
    total_weight_kg,
)
from qf.domain.serialization import (
    TargetParseError,
    parse_target,
    parse_target_lenient,
    serialize_target,
)
from qf.domain.tasks import SHIPMENT_EXTRACTION, TASKS, TaskSpec, get_task, register_task
from qf.domain.units import (
    LB_TO_KG,
    T_TO_KG,
    format_kg,
    from_kg,
    normalize_number,
    render_value_from_lbs,
    to_kg,
)

__all__ = [
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
