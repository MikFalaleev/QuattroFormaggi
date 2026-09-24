"""Evaluation types (plan step 9): the score of one answer, shared by every metric.

`score_prediction` parses a model answer once (strictly and leniently) and fills a `CaseScore`;
metrics only read it. A card_v2 answer gets a `CaseScoreV2`: the same score plus its special
conditions (sub-step V6), so the scores of card_v1 runs keep their exact form. See
docs/EVAL_SPEC.md for the definitions.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from qf.contracts._model import ContractModel, NonEmptyStr
from qf.contracts.card_v2 import ConditionKind
from qf.contracts.rendering import AnyFieldName

__all__ = ["CaseScore", "CaseScoreV2", "MetricValue"]

MetricValue = float | int | bool | tuple[float, ...] | None
"""A metric's value for one case: a number, a flag, a tuple of counts, or None (not applicable)."""

Count = Annotated[int, Field(ge=0)]


class CaseScore(ContractModel):
    record_id: NonEmptyStr
    json_parsed: bool  # the output is one bare, well-formed JSON object (strict format)
    schema_valid: bool  # ...and a valid answer of the record's schema: this is json_valid_rate
    json_parsed_lenient: bool  # valid after stripping spaces / one code fence: fields scored
    generation_error: str | None  # timeout, HTTP error etc. of the backend
    field_correct: dict[AnyFieldName, bool | None]  # None: gold and answer both null / empty
    key_fields_correct: bool | None  # None: some key field of the gold is null
    missing_tp: Count
    missing_fp: Count
    missing_fn: Count
    missing_exact: bool
    conflict_detected: bool | None  # None: the gold has no conflicts
    hallucinated_fields: list[AnyFieldName]  # gold null (no conflict), answer not null
    critical_errors: list[str]
    notes: list[str]  # non-critical diagnostics, e.g. computed_weight_total

    @property
    def failed(self) -> bool:
        """No answer could be scored: a generation error or not even the lenient parse."""
        return self.generation_error is not None or not self.json_parsed_lenient


class CaseScoreV2(CaseScore):
    """The score of a card_v2 answer. Conditions are compared by kind: `condition_correct` is
    None where neither the gold nor the answer has the kind, True where both have it with the
    same values (lists as sets, dimensions in metres within 0.01 m)."""

    conditions_gold: list[ConditionKind]
    conditions_answer: list[ConditionKind]  # [] for a failed answer
    condition_correct: dict[ConditionKind, bool | None]
