"""Contract test of every registered template family x hard case (plan C.8, invariants E1-E15).

A new family or hard case registered in `TEMPLATE_FAMILIES` / `HARD_CASES` is checked here
without writing new tests. The dataset-level invariants E9 and E10 are in `test_generate.py`.
"""

from __future__ import annotations

from datetime import timedelta
from functools import cache
from itertools import combinations
from random import Random

import pytest

from qf.contracts import RequestDraft, TemplateFamily
from qf.data import HARD_CASES, TEMPLATE_FAMILIES, GeneratedRecord, render_record
from qf.data.render import DROPPABLE_FIELDS, LayoutFamily
from qf.domain import compute_missing_fields
from tests.generation import INVARIANTS, SOURCE_PREFIX, facts_of, money_of, repo_config, violations

FAMILIES = TEMPLATE_FAMILIES.names()
KINDS = ["clean", *HARD_CASES.names()]
RECORDS_PER_CASE = 40


def _family(name: str) -> TemplateFamily:
    return TEMPLATE_FAMILIES.get(name)()


@cache
def generated(family: str, kind: str, count: int = RECORDS_PER_CASE) -> tuple[GeneratedRecord, ...]:
    cfg = repo_config()
    facts = facts_of()
    ood = "family" if _family(family).ood_only else None
    return tuple(
        render_record(
            facts[i % len(facts)],
            "test_ood" if ood else "train",
            i,
            cfg,
            ood_reason=ood,
            source_prefix=SOURCE_PREFIX,
            family=family,
            kind=kind,
        )  # fmt: skip
        for i in range(count)
    )


@pytest.mark.parametrize("invariant", sorted(INVARIANTS))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("family", FAMILIES)
def test_invariant(family: str, kind: str, invariant: str) -> None:
    facts = {f.load_id: f for f in facts_of()}
    assert violations(invariant, generated(family, kind), facts, money_of()) == []


@pytest.mark.parametrize("kind", HARD_CASES.names())
@pytest.mark.parametrize("family", FAMILIES)
def test_hard_case_is_applied(family: str, kind: str) -> None:
    applied = [g for g in generated(family, kind) if kind in g.draft.hard_cases]
    assert len(applied) == RECORDS_PER_CASE
    assert all(g.record.variant.hard_cases == [kind] for g in applied)


@pytest.mark.parametrize("family", FAMILIES)
def test_family_attributes(family: str) -> None:
    template = _family(family)
    assert template.name == family
    assert isinstance(template, TemplateFamily)
    assert template.ood_only == (family in ("T7", "T8"))
    assert isinstance(template, LayoutFamily) and len(template.orders) >= 5


@pytest.mark.parametrize("family", FAMILIES)
def test_every_drop_subset_is_rendered(family: str) -> None:
    """Every subset of 1-3 dropped fields (129) renders, and exactly those fields are null."""
    template = _family(family)
    facts = facts_of()
    subsets = [s for k in (1, 2, 3) for s in combinations(DROPPABLE_FIELDS, k)]
    assert len(subsets) == 129
    for i, subset in enumerate(subsets):
        f = facts[i % len(facts)]
        draft = RequestDraft(
            language=template.language, request_date=f.pickup_date - timedelta(days=3),
            weight_unit="kg", weight_mode="total", date_style="text",
            city_lang=template.language, dropped_fields=list(subset), conflicts={},
            distractors={}, hard_cases=["dropped_fields"], ood_reason=None,
        )  # fmt: skip
        rendered = template.render(f, draft, Random(i))
        card = rendered.target.card
        null = {name for name, value in card if value is None}
        assert null == {*subset, "weight_per_piece", "temperature_c"}, (family, subset)
        assert rendered.target.missing_fields == compute_missing_fields(card)
        assert all(s in rendered.text for found in rendered.evidence.values() for s in found)
