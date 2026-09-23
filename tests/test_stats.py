"""Bootstrap intervals, paired differences and McNemar's test (plan step 9)."""

from __future__ import annotations

import pytest

from qf.eval import (
    bootstrap_ci,
    mcnemar,
    paired_bootstrap_diff,
    percentile,
    resample_indices,
    statistic_ci,
)


def test_bootstrap_ci_contains_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    low, high = bootstrap_ci(values, n=1000, seed=3)
    assert low <= sum(values) / len(values) <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_is_reproducible_and_rejects_empty() -> None:
    values = [0.2, 0.4, 0.9, 0.1]
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)
    assert bootstrap_ci([0.5] * 5) == (0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_resamples_are_shared_and_seeded() -> None:
    assert resample_indices(5, n=3, seed=0) == resample_indices(5, n=3, seed=0)
    assert resample_indices(5, n=3, seed=0) != resample_indices(5, n=3, seed=1)
    assert all(len(s) == 5 and all(0 <= i < 5 for i in s) for s in resample_indices(5, n=3))
    assert resample_indices(0) == []


def test_percentile_nearest_rank() -> None:
    ordered = [1.0, 2.0, 3.0, 4.0]
    assert percentile(ordered, 0.0) == 1.0
    assert percentile(ordered, 0.5) == 2.0
    assert percentile(ordered, 1.0) == 4.0


def test_statistic_ci_none_when_never_applicable() -> None:
    values: list[float | None] = [None, None]
    assert statistic_ci(values, lambda _: None, resample_indices(2, n=10)) is None


def test_paired_diff_zero_for_identical() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert paired_bootstrap_diff(values, values, n=500) == (0.0, 0.0, 0.0)


def test_paired_diff_sign_and_shared_indices() -> None:
    a = [0.0] * 10
    b = [1.0] * 5 + [0.0] * 5
    diff, low, high = paired_bootstrap_diff(a, b, n=500) or (0.0, 0.0, 0.0)
    assert diff == 0.5 and 0.0 <= low <= 0.5 <= high <= 1.0
    indices = resample_indices(10, n=500, seed=7)
    assert paired_bootstrap_diff(a, b, indices=indices) == paired_bootstrap_diff(
        a, b, indices=indices
    )
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 0.0])


def test_paired_diff_none_when_statistic_does_not_apply() -> None:
    assert paired_bootstrap_diff([1.0], [0.0], statistic=lambda _: None) is None


def test_mcnemar_known_values() -> None:
    # 5 cases fixed, none broken: p = 2 * 0.5**5
    assert mcnemar([False] * 5 + [True] * 3, [True] * 8) == (5, 0, 0.0625)
    # 6 fixed, 2 broken: p = 2 * (C(8,0) + C(8,1) + C(8,2)) / 2**8 = 2 * 37 / 256
    a = [False] * 6 + [True] * 2 + [True] * 4
    b = [True] * 6 + [False] * 2 + [True] * 4
    assert mcnemar(a, b) == (6, 2, pytest.approx(74 / 256))
    # balanced: the p-value is capped at 1
    assert (
        mcnemar([False, False, False, True, True, True], [True, True, True] + [False] * 3)[2] == 1.0
    )


def test_mcnemar_without_discordant_pairs() -> None:
    assert mcnemar([True, False, True], [True, False, True]) == (0, 0, 1.0)
    assert mcnemar([], []) == (0, 0, 1.0)
    with pytest.raises(ValueError):
        mcnemar([True], [])
