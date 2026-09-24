"""The metrics of v0.1 (plan step 9, docs/EVAL_SPEC.md). Each is a `Metric` registered by name.

Rates are means over the cases where the metric applies (value not None); counts are sums.
A case whose answer could not be scored counts as a failure for rates of correctness and as
"not assessable" (None) for hallucination and critical errors. The metrics of the special
conditions (sub-step V6) apply only to card_v2 scores (`CaseScoreV2`), None for card_v1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar

from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    AnyFieldName,
    CaseScore,
    CaseScoreV2,
    ConditionKind,
    MetricValue,
)
from qf.eval.metrics.case_scoring import FIELDS
from qf.eval.metrics.case_scoring_v2 import CONDITIONS_FIELD, CRITICAL_ERRORS_V2, FIELDS_V2
from qf.eval.metrics.registry import METRICS

__all__ = ["ConditionDetection", "CountMetric", "FieldAccuracyMicro", "MissingF1", "RateMetric"]


def _scalars(values: Sequence[MetricValue]) -> list[float]:
    """Numbers and flags of the cases where the metric applies (None and tuples dropped)."""
    return [float(v) for v in values if isinstance(v, int | float)]


class RateMetric:
    """Mean of a per-case flag or number over the cases where it applies."""

    name: ClassVar[str]
    higher_is_better: ClassVar[bool] = True
    extract: ClassVar[Callable[[CaseScore], MetricValue]]

    def value(self, case: CaseScore) -> MetricValue:
        return type(self).extract(case)

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        present = _scalars(values)
        return sum(present) / len(present) if present else None


class CountMetric(RateMetric):
    """Sum of a per-case count (lower is better)."""

    higher_is_better: ClassVar[bool] = False

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        present = _scalars(values)
        return float(sum(present)) if present else None


def _rate(name: str, extract: Callable[[CaseScore], MetricValue], *, higher: bool = True) -> None:
    cls = type(f"Rate_{name}", (RateMetric,), {
        "name": name, "higher_is_better": higher, "extract": staticmethod(extract),
    })  # fmt: skip
    METRICS.register(name)(cls)


def _count(name: str, extract: Callable[[CaseScore], MetricValue]) -> None:
    cls = type(f"Count_{name}", (CountMetric,), {"name": name, "extract": staticmethod(extract)})
    METRICS.register(name)(cls)


_rate("json_valid_rate", lambda c: c.schema_valid)
_rate("lenient_parse_rate", lambda c: c.json_parsed_lenient)
_rate("failed_output_rate", lambda c: c.failed, higher=False)
_rate("key_field_accuracy", lambda c: c.key_fields_correct)
_rate("missing_exact_rate", lambda c: c.missing_exact)
_rate("conflict_recall", lambda c: c.conflict_detected)
_rate("hallucination_rate", lambda c: None if c.failed else bool(c.hallucinated_fields),
      higher=False)  # fmt: skip
_count("critical_error_count", lambda c: None if c.failed else len(c.critical_errors))


def _field_extract(field: AnyFieldName) -> Callable[[CaseScore], MetricValue]:
    return lambda c: c.field_correct.get(field)


def _critical_extract(kind: str) -> Callable[[CaseScore], MetricValue]:
    return lambda c: None if c.failed else c.critical_errors.count(kind)


def _condition_extract(kind: ConditionKind) -> Callable[[CaseScore], MetricValue]:
    return lambda c: c.condition_correct.get(kind) if isinstance(c, CaseScoreV2) else None


def _conditions_exact(case: CaseScore) -> MetricValue:
    """All conditions right, including "none" (a failed answer is wrong); card_v2 only."""
    if not isinstance(case, CaseScoreV2):
        return None
    return not case.failed and case.field_correct.get(CONDITIONS_FIELD) is not False


# card_v1 fields first, then the fields only card_v2 has (the names of card_v1 stay as they were)
for _field in (*FIELDS, *(f for f in FIELDS_V2 if f not in FIELDS)):
    _rate(f"field_accuracy.{_field}", _field_extract(_field))
for _kind in CRITICAL_ERRORS_V2:
    _count(f"critical_error_count.{_kind}", _critical_extract(_kind))
_rate("special_conditions_exact_rate", _conditions_exact)
for _condition in CARD_V2_CONDITION_KINDS:
    _rate(f"field_accuracy.{CONDITIONS_FIELD}.{_condition}", _condition_extract(_condition))


@METRICS.register("field_accuracy_micro")
class FieldAccuracyMicro:
    """Correct fields / scored fields, pooled over all cases."""

    name = "field_accuracy_micro"
    higher_is_better = True

    def value(self, case: CaseScore) -> MetricValue:
        scored = [v for v in case.field_correct.values() if v is not None]
        return (float(sum(scored)), float(len(scored)))

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        pairs = [v for v in values if isinstance(v, tuple)]
        total = sum(pair[1] for pair in pairs)
        return sum(pair[0] for pair in pairs) / total if total else None


@METRICS.register("missing_f1")
class MissingF1:
    """F1 of the predicted missing_fields against the gold, pooled over cases (2TP/(2TP+FP+FN));
    1.0 when neither the gold nor the answers list anything."""

    name = "missing_f1"
    higher_is_better = True

    def value(self, case: CaseScore) -> MetricValue:
        return (float(case.missing_tp), float(case.missing_fp), float(case.missing_fn))

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        triples = [v for v in values if isinstance(v, tuple)]
        if not triples:
            return None
        tp, fp, fn = (sum(t[i] for t in triples) for i in range(3))
        return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


class ConditionDetection:
    """Whether the kinds of the special conditions are found, pooled over cases: precision
    TP/(TP+FP) or recall TP/(TP+FN) of the kinds (their values are
    `field_accuracy.special_conditions.<kind>`), of all kinds or of one. card_v2 only."""

    name: ClassVar[str]
    higher_is_better: ClassVar[bool] = True
    kind: ClassVar[ConditionKind | None]
    recall: ClassVar[bool]

    def value(self, case: CaseScore) -> MetricValue:
        if not isinstance(case, CaseScoreV2):
            return None
        wanted = {type(self).kind} if type(self).kind else set(CARD_V2_CONDITION_KINDS)
        gold = set(case.conditions_gold) & wanted
        answer = set(case.conditions_answer) & wanted
        wrong = gold - answer if type(self).recall else answer - gold
        return (float(len(gold & answer)), float(len(wrong)))

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        pairs = [v for v in values if isinstance(v, tuple)]
        found, total = sum(p[0] for p in pairs), sum(p[0] + p[1] for p in pairs)
        return found / total if total else None


def _detection(name: str, kind: ConditionKind | None, recall: bool) -> None:
    cls = type(f"Detection_{name}", (ConditionDetection,), {
        "name": name, "kind": kind, "recall": recall,
    })  # fmt: skip
    METRICS.register(name)(cls)


for _measure, _recall in (("precision", False), ("recall", True)):
    _detection(f"condition_{_measure}", None, _recall)
    for _condition in CARD_V2_CONDITION_KINDS:
        _detection(f"condition_{_measure}.{_condition}", _condition, _recall)
