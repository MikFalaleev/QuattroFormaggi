"""Contract test of every registered template family x hard case (plan C.8, invariants E1-E17).

A new family or hard case registered in `TEMPLATE_FAMILIES` / `HARD_CASES` is checked here
without writing new tests: card_v1 with the cases that apply to card_v1 facts, card_v2 with
every case on card_v2 facts that it applies to (sub-step V3). The dataset-level invariants E9
and E10 are in `test_generate.py`.
"""

from __future__ import annotations

from datetime import timedelta
from functools import cache
from itertools import combinations
from random import Random

import pytest

from qf.contracts import AnyLoadFacts, RequestDraft, TemplateFamily
from qf.data import HARD_CASES, TEMPLATE_FAMILIES, GenerateConfig, GeneratedRecord, render_record
from qf.data.render import DROPPABLE_FIELDS, LayoutFamily, RenderError
from qf.domain import compute_missing_fields
from tests.generation import (
    INVARIANTS,
    SOURCE_PREFIX,
    facts_of,
    money_of,
    repo_config,
    repo_config_v2,
    synthetic_facts_v2,
    violations,
)

FAMILIES = TEMPLATE_FAMILIES.names()
V1_CASES = [name for name in HARD_CASES.names()
            if any(HARD_CASES.get(name)().applicable(f) for f in facts_of())]  # fmt: skip
KINDS = ["clean", *V1_CASES]
KINDS_V2 = ["clean", *HARD_CASES.names()]
RECORDS_PER_CASE = 40


def _family(name: str) -> TemplateFamily:
    return TEMPLATE_FAMILIES.get(name)()


def _render_all(family: str, kind: str, cfg: GenerateConfig, facts: tuple[AnyLoadFacts, ...],
                count: int) -> tuple[GeneratedRecord, ...]:  # fmt: skip
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


@cache
def generated(family: str, kind: str, count: int = RECORDS_PER_CASE) -> tuple[GeneratedRecord, ...]:
    return _render_all(family, kind, repo_config(), facts_of(), count)


@cache
def generated_v2(family: str, kind: str,
                 count: int = RECORDS_PER_CASE) -> tuple[GeneratedRecord, ...]:  # fmt: skip
    """card_v2 records on the facts the case applies to (all facts for "clean")."""
    facts = synthetic_facts_v2()
    if kind != "clean":
        facts = tuple(f for f in facts if HARD_CASES.get(kind)().applicable(f))
    return _render_all(family, kind, repo_config_v2(), facts, count)


@pytest.mark.parametrize("invariant", sorted(INVARIANTS))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("family", FAMILIES)
def test_invariant(family: str, kind: str, invariant: str) -> None:
    facts = {f.load_id: f for f in facts_of()}
    assert violations(invariant, generated(family, kind), facts, money_of()) == []


@pytest.mark.parametrize("invariant", sorted(INVARIANTS))
@pytest.mark.parametrize("kind", KINDS_V2)
@pytest.mark.parametrize("family", FAMILIES)
def test_invariant_v2(family: str, kind: str, invariant: str) -> None:
    facts = {f.load_id: f for f in synthetic_facts_v2()}
    assert violations(invariant, generated_v2(family, kind), facts, {}) == []


@pytest.mark.parametrize("kind", V1_CASES)
@pytest.mark.parametrize("family", FAMILIES)
def test_hard_case_is_applied(family: str, kind: str) -> None:
    applied = [g for g in generated(family, kind) if kind in g.draft.hard_cases]
    assert len(applied) == RECORDS_PER_CASE
    assert all(g.record.variant.hard_cases == [kind] for g in applied)


@pytest.mark.parametrize("kind", HARD_CASES.names())
@pytest.mark.parametrize("family", FAMILIES)
def test_hard_case_is_applied_v2(family: str, kind: str) -> None:
    applied = [g for g in generated_v2(family, kind) if kind in g.draft.hard_cases]
    assert len(applied) == RECORDS_PER_CASE
    assert all(g.record.schema_version == "card_v2" for g in applied)


def test_condition_cases_do_not_apply_to_card_v1_facts() -> None:
    assert set(HARD_CASES.names()) - set(V1_CASES) == {
        "condition_no_values", "required_condition_dropped", "oversize_partial",
        "conditions_scattered",
    }  # fmt: skip


@pytest.mark.parametrize("family", FAMILIES)
def test_versions_do_not_mix(family: str) -> None:
    template = _family(family)
    v1_facts, v2_facts = facts_of()[0], synthetic_facts_v2()[0]
    draft = RequestDraft(
        language=template.language, request_date=v1_facts.pickup_date, weight_unit="kg",
        weight_mode="total", date_style="iso", city_lang=template.language, dropped_fields=[],
        conflicts={}, distractors={}, hard_cases=[], ood_reason=None,
    )  # fmt: skip
    with pytest.raises(RenderError, match="card_v1 draft cannot render LoadFactsV2"):
        template.render(v2_facts, draft, Random(0))
    v2_draft = draft.model_copy(update={"schema_version": "card_v2"})
    with pytest.raises(RenderError, match="card_v2 draft cannot render LoadFacts"):
        template.render(v1_facts, v2_draft, Random(0))


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
