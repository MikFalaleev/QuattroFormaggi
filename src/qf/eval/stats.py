"""Statistics of evaluation results (plan step 9): bootstrap intervals, paired differences,
McNemar's exact test. Pure Python with a seeded RNG, so every number is reproducible.

Percentile intervals use the nearest-rank method. With few cases (8 per hard case of bench_v1)
the intervals are wide; the report shows them rather than hiding them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from random import Random
from typing import TypeVar

__all__ = [
    "bootstrap_ci",
    "mcnemar",
    "paired_bootstrap_diff",
    "percentile",
    "resample_indices",
    "statistic_ci",
]

T = TypeVar("T")
Statistic = Callable[[Sequence[T]], float | None]


def resample_indices(size: int, n: int = 2000, seed: int = 0) -> list[list[int]]:
    """`n` bootstrap resamples of `size` case positions, drawn once and shared by all metrics
    of a run, so their intervals come from the same resamples."""
    rng = Random(seed)
    return [[rng.randrange(size) for _ in range(size)] for _ in range(n)] if size else []


def percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of a sorted, non-empty sequence (q in [0, 1])."""
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def statistic_ci(
    values: Sequence[T], statistic: Statistic[T], indices: Sequence[Sequence[int]],
    alpha: float = 0.05,
) -> tuple[float, float] | None:  # fmt: skip
    """Percentile interval of `statistic` over the given resamples; None if it never applies."""
    results = sorted(
        value for sample in indices if (value := statistic([values[i] for i in sample])) is not None
    )
    if not results:
        return None
    return percentile(results, alpha / 2), percentile(results, 1 - alpha / 2)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    values: Sequence[float], n: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """95% (by default) percentile interval of the mean."""
    if not values:
        raise ValueError("bootstrap of an empty sample")
    interval = statistic_ci(values, _mean, resample_indices(len(values), n, seed), alpha)
    assert interval is not None
    return interval


def paired_bootstrap_diff(
    a: Sequence[T], b: Sequence[T], n: int = 2000, seed: int = 0, alpha: float = 0.05,
    statistic: Statistic[T] | None = None, indices: Sequence[Sequence[int]] | None = None,
) -> tuple[float, float, float] | None:  # fmt: skip
    """(statistic(b) - statistic(a), interval low, high) over paired resamples of the same
    cases; the default statistic is the mean. `indices` (from `resample_indices`) replaces
    `n` and `seed`, so several metrics can share one set of resamples. None if the statistic
    does not apply to one of the samples."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    stat: Statistic[T] = statistic or _mean  # type: ignore[assignment]
    base_a, base_b = stat(a), stat(b)
    if base_a is None or base_b is None:
        return None
    diffs = []
    for sample in indices if indices is not None else resample_indices(len(a), n, seed):
        sa, sb = stat([a[i] for i in sample]), stat([b[i] for i in sample])
        if sa is not None and sb is not None:
            diffs.append(sb - sa)
    if not diffs:
        return None
    diffs.sort()
    return base_b - base_a, percentile(diffs, alpha / 2), percentile(diffs, 1 - alpha / 2)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """(b01, b10, p): b01 cases wrong in A and right in B, b10 the reverse; p is the exact
    two-sided binomial p-value on the discordant pairs (1.0 when there are none)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    b01 = sum(1 for x, y in zip(a, b, strict=True) if not x and y)
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    discordant = b01 + b10
    if discordant == 0:
        return b01, b10, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b01, b10) + 1)) / 2**discordant
    return b01, b10, min(1.0, 2 * tail)
