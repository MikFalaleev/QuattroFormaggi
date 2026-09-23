"""Hand-written benchmark cases (plan step 8): YAML written by a person -> SFT records.

A person writes the request text and the correct card (and conflicts, if any); code builds the
messages, computes `missing_fields` by the rule and writes the canonical answer, so a manual
case goes through the same checks as a generated one. Format (a YAML list):

    - author: ivanov            # latin letters, digits, _ or -
      language: ru              # ru | en
      request_date: 2026-09-20
      text: |
        Добрый день! Нужна машина из Казани в Москву ...
      card:
        shipper_name: ООО «Ромашка»
        ...all 11 card fields, null when the text does not say...
      conflicts: []             # optional: [{field: pieces, values: [18, 20]}]
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

from qf.common import DataValidationError, Issue, QFError, StrictConfig, format_validation_error
from qf.contracts import (
    CARD_SCHEMA_VERSION,
    Conflict,
    ExtractionTarget,
    Message,
    SFTRecord,
    ShipmentCard,
    VariantInfo,
)
from qf.domain import (
    CITIES,
    SHIPMENT_EXTRACTION,
    build_messages,
    compute_missing_fields,
    serialize_target,
)

__all__ = ["ManualCase", "load_manual_cases"]


class ManualCase(StrictConfig):
    author: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    language: Literal["ru", "en"]
    request_date: date
    text: str = Field(min_length=1)
    card: dict[str, Any]
    conflicts: list[dict[str, Any]] = Field(default_factory=list)


def _place_warnings(card: ShipmentCard, case_id: str) -> tuple[list[Issue], list[str]]:
    """A catalog city must carry its catalog region; other cities are allowed with a warning."""
    errors, warnings = [], []
    for name in ("origin", "destination"):
        place = getattr(card, name)
        if place is None:
            continue
        info = CITIES.get(place.city)
        if info is None:
            warnings.append(f"{case_id}: {name} {place.city!r} is not one of the 20 catalog cities")
        elif info.region != place.region:
            message = f"{place.city} is in {info.region!r}, not {place.region!r}"
            errors.append(Issue(case_id, "bench", "REGION", message))
    return errors, warnings


def _record(case: ManualCase, n: int) -> tuple[SFTRecord, list[Issue], list[str]]:
    case_id = f"manual-{case.author}-{n}"
    # Through JSON: a date written as 2026-09-22 or '2026-09-22' is an ISO string there, while
    # "18" stays a string and is refused as a number (strict contracts, D-041).
    card = ShipmentCard.model_validate_json(json.dumps(case.card, default=str))
    conflicts = [Conflict.model_validate_json(json.dumps(c, default=str)) for c in case.conflicts]
    target = ExtractionTarget(card=card, missing_fields=compute_missing_fields(card),
                              conflicts=conflicts)  # fmt: skip
    errors, warnings = _place_warnings(card, case_id)
    weight = card.weight_total or card.weight_per_piece
    mode: Literal["total", "per_piece", "none"] = "none"
    if card.weight_total is not None:
        mode = "total"
    elif card.weight_per_piece is not None:
        mode = "per_piece"
    task = SHIPMENT_EXTRACTION
    record = SFTRecord(
        id=case_id, task=task.name, group_id=f"manual:{case.author}:{n}",
        source_id=f"manual:{case.author}:{n}", language=case.language, reviewed=True,
        synthetic=False, template_family="manual",
        variant=VariantInfo(weight_unit=weight.unit if weight else None, weight_mode=mode,
                            dropped_fields=[], hard_cases=[], request_date=case.request_date,
                            date_style="text", city_lang=case.language, ood_reason=None),
        schema_version=CARD_SCHEMA_VERSION, split="bench",
        messages=[*build_messages(case.text.strip(), case.request_date, case.language,
                                  task.system_prompt_version),
                  Message(role="assistant", content=serialize_target(target))],
    )  # fmt: skip
    return record, errors, warnings


def load_manual_cases(path: Path) -> tuple[list[SFTRecord], list[str]]:
    """Records of the YAML file and warnings; every invalid case is reported at once."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError) as exc:
        raise QFError(f"cannot read manual cases {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise QFError(f"{path}: the top level must be a list of cases")
    records, issues, warnings = [], [], []
    numbers: Counter[str] = Counter()
    for index, item in enumerate(raw, start=1):
        try:
            case = ManualCase.model_validate(item)
            numbers[case.author] += 1
            record, errors, notes = _record(case, numbers[case.author])
        except ValidationError as exc:
            message = format_validation_error(exc)
            issues.append(Issue(f"{path.name}#{index}", "bench", "SCHEMA", message))
            continue
        records.append(record)
        issues += errors
        warnings += notes
    if issues:
        raise DataValidationError(issues)
    return records, warnings
