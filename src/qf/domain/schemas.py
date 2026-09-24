"""Registry of answer schemas by version (D-086): one entry per `TargetSchemaVersion`.

Code that handles records of several schema versions (record checks, later the generator and
the benchmark) looks the schema up by `SFTRecord.schema_version` instead of branching on the
version. An entry bundles what the domain layer knows about a schema: parsing, the canonical
serialization, the missing-fields rule, consistency checks and the JSON schema. Field
comparison and critical-error detectors belong to `qf.eval` and get their own registry by
version there (sub-step V6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, get_args

from pydantic import BaseModel

from qf.common import Registry
from qf.contracts import (
    CARD_SCHEMA_VERSION,
    CARD_V2_SCHEMA_VERSION,
    ExtractionTarget,
    ExtractionTargetV2,
    FieldName,
    FieldNameV2,
    ShipmentCard,
    ShipmentCardV2,
    TargetSchemaVersion,
    require_supported,
)
from qf.domain.rules import check_target_consistency, compute_missing_fields
from qf.domain.rules_v2 import (
    canonical_card_v2,
    check_target_consistency_v2,
    compute_missing_fields_v2,
)
from qf.domain.serialization import (
    parse_target,
    parse_target_lenient,
    parse_target_lenient_v2,
    parse_target_v2,
    serialize_target,
)

__all__ = ["TARGET_SCHEMAS", "TargetSchema", "get_target_schema", "register_target_schema"]

TargetT = TypeVar("TargetT", bound=BaseModel)


@dataclass(frozen=True)
class TargetSchema(Generic[TargetT]):
    """What the domain layer knows about one answer schema version."""

    version: TargetSchemaVersion
    target_model: type[TargetT]
    card_model: type[BaseModel]  # the `card` of the answer
    card_fields: tuple[str, ...]  # card field names in declaration order
    parse: Callable[[str], TargetT]  # strict (`TargetParseError`)
    parse_lenient: Callable[[str], TargetT]  # surrounding whitespace and one code fence
    serialize: Callable[[TargetT], str]  # the canonical assistant text
    missing_fields: Callable[[Any], Sequence[str]]  # the rule, applied to a card
    check_consistency: Callable[[TargetT], list[str]]  # rule violations of an answer
    canonicalize: Callable[[Any], Any]  # a card in the canonical order of its lists

    def json_schema(self) -> dict[str, Any]:
        """JSON schema of the whole answer, for constrained decoding (step 10 checks whether
        the runtime's grammar accepts it)."""
        return self.target_model.model_json_schema()


TARGET_SCHEMAS: Registry[TargetSchema[Any]] = Registry("target_schema")


def _same(card: Any) -> Any:
    return card  # card_v1 has no lists whose order is a rule


def register_target_schema(schema: TargetSchema[Any]) -> TargetSchema[Any]:
    require_supported("target", schema.version)
    TARGET_SCHEMAS.register(schema.version)(schema)
    return schema


def get_target_schema(version: str) -> TargetSchema[Any]:
    """The registered schema; QFError listing the known versions otherwise."""
    return TARGET_SCHEMAS.get(version)


register_target_schema(
    TargetSchema(
        version=CARD_SCHEMA_VERSION,
        target_model=ExtractionTarget,
        card_model=ShipmentCard,
        card_fields=get_args(FieldName),
        parse=parse_target,
        parse_lenient=parse_target_lenient,
        serialize=serialize_target,
        missing_fields=compute_missing_fields,
        check_consistency=check_target_consistency,
        canonicalize=_same,
    )
)
register_target_schema(
    TargetSchema(
        version=CARD_V2_SCHEMA_VERSION,
        target_model=ExtractionTargetV2,
        card_model=ShipmentCardV2,
        card_fields=get_args(FieldNameV2),
        parse=parse_target_v2,
        parse_lenient=parse_target_lenient_v2,
        serialize=serialize_target,
        missing_fields=compute_missing_fields_v2,
        check_consistency=check_target_consistency_v2,
        canonicalize=canonical_card_v2,
    )
)
