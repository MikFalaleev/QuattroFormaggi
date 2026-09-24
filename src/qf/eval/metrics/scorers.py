"""Registry of answer scorers by schema version (D-086, D-087; sub-step V6).

The eval side of the schema registry `qf.domain.TARGET_SCHEMAS`: how an answer of the schema is
scored, which score model its `scores.jsonl` holds, its card fields and critical errors, and
the columns of the critical-error table of the report. The harness and the report look the
scorer up by `SFTRecord.schema_version` instead of branching on the version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qf.common import Registry
from qf.contracts import (
    CARD_SCHEMA_VERSION,
    CARD_V2_SCHEMA_VERSION,
    AnyFieldName,
    CaseScore,
    CaseScoreV2,
    TargetSchemaVersion,
)
from qf.eval.metrics.case_scoring import CRITICAL_ERRORS, FIELDS, score_prediction
from qf.eval.metrics.case_scoring_v2 import CRITICAL_ERRORS_V2, FIELDS_V2, score_prediction_v2

__all__ = ["SCORERS", "KindColumn", "ScoreFunction", "Scorer", "get_scorer"]


class ScoreFunction(Protocol):
    def __call__(self, record_id: str, gold: Any, raw_output: str, *,
                 generation_error: str | None = None) -> CaseScore: ...  # fmt: skip


KindColumn = tuple[str, frozenset[str] | None]
"""A column of the critical-error table: its title and the kinds of cases it sums (None: all)."""


@dataclass(frozen=True)
class Scorer:
    version: TargetSchemaVersion
    score: ScoreFunction  # (record id, parsed gold, raw answer) -> score
    score_model: type[CaseScore]  # the lines of scores.jsonl
    fields: tuple[AnyFieldName, ...]  # card fields in declaration order
    critical_errors: tuple[str, ...]
    kind_columns: tuple[KindColumn, ...]
    kinds_note: str  # the explanation above the critical-error table


SCORERS: Registry[Scorer] = Registry("scorer")


def get_scorer(version: str) -> Scorer:
    """The registered scorer; QFError listing the scored versions otherwise."""
    return SCORERS.get(version)


SCORERS.register(CARD_SCHEMA_VERSION)(Scorer(
    version=CARD_SCHEMA_VERSION, score=score_prediction, score_model=CaseScore, fields=FIELDS,
    critical_errors=CRITICAL_ERRORS,
    kind_columns=(("clean", frozenset({"clean"})), ("missing", frozenset({"missing"})),
                  ("hard", frozenset({"hard"})),
                  ("трудные (missing + hard)", frozenset({"missing", "hard"})), ("всего", None)),
    kinds_note="clean — чистые заявки, missing — выпавшие поля, hard — остальные трудные "
               "случаи; трудный срез gates — missing + hard. Неуспешные ответы не оцениваются "
               "(строка «не оценено»).",
))  # fmt: skip
SCORERS.register(CARD_V2_SCHEMA_VERSION)(Scorer(
    version=CARD_V2_SCHEMA_VERSION, score=score_prediction_v2, score_model=CaseScoreV2,
    fields=FIELDS_V2, critical_errors=CRITICAL_ERRORS_V2,
    kind_columns=(("clean", frozenset({"clean"})), ("conditions", frozenset({"conditions"})),
                  ("missing", frozenset({"missing"})), ("hard", frozenset({"hard"})),
                  ("manual", frozenset({"manual"})),
                  ("трудные (всё, кроме clean)",
                   frozenset({"conditions", "missing", "hard", "manual"})),
                  ("всего", None)),
    kinds_note="clean — чистые заявки без особых условий, conditions — с особыми условиями, "
               "missing — выпавшие поля, hard — остальные трудные случаи, manual — ручные "
               "заявки и заявки-моки (D-103); трудный срез gates — всё, кроме clean (D-085; "
               "входит ли в него manual — решение шага 10). Неуспешные ответы не оцениваются "
               "(строка «не оценено»).",
))  # fmt: skip
