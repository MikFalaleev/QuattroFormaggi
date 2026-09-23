"""Layouts, the gold-answer assembler and the base class of template families (plan 6.3, 6.5).

A family only composes fragments: it lists slots (sentences with alternatives) and orders of
slots. `fill_layout` places every fragment exactly once and `assemble` builds the gold answer
from the evidence that ended up in the text. A layout that loses a field is an error, never a
silently wrong gold answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from pydantic import BaseModel

from qf.common import QFError
from qf.contracts import (
    Conflict,
    ExtractionTarget,
    FieldName,
    Language,
    LoadFacts,
    RenderedField,
    RenderedRequest,
    RequestDraft,
    ShipmentCard,
    VariantInfo,
)
from qf.data.render.fields import Fragment, build_fragments
from qf.domain import compute_missing_fields

__all__ = [
    "HardCaseNotApplicable",
    "Layout",
    "LayoutFamily",
    "RenderError",
    "Slot",
    "assemble",
    "fill_layout",
]

Slot = tuple[str, ...]
"""Alternative sentences; `{placeholder}` names a fragment."""
Layout = tuple[Slot, ...]

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_DOUBLE_DOT = re.compile(r"(?<!\.)\.\.(?!\.)")


class RenderError(QFError):
    """A family or layout cannot express the draft (a bug in templates, not in data)."""


class HardCaseNotApplicable(QFError):
    """The hard case cannot be applied to this load; the generator picks another one."""


def _placeholders(template: str) -> list[str]:
    return _PLACEHOLDER.findall(template)


def fill_layout(
    layout: Layout,
    fragments: Mapping[str, Fragment],
    rng: Random,
    joiner: str,
    *,
    capitalize: bool = True,
) -> tuple[str, list[RenderedField]]:
    """Text of the layout and the evidence placed in it, in text order.

    In every slot the chosen sentence must cover all coverage keys of the slot that are still
    present; sentences whose placeholders are missing are skipped. Afterwards every coverage
    key of every fragment must be placed, otherwise `RenderError`. With `capitalize` every
    sentence starts with a capital letter; a sentence ending with a line break is followed by
    the next one without `joiner`.
    """
    required = frozenset[str]().union(*(fragment.covers for fragment in fragments.values()))
    placed: set[str] = set()
    parts: list[str] = []
    evidence: list[RenderedField] = []
    for slot in layout:
        available = [alt for alt in slot if all(p in fragments for p in _placeholders(alt))]
        need = (
            frozenset[str]().union(
                *(fragments[p].covers for alt in available for p in _placeholders(alt))
            )
            - placed
        )
        candidates = [
            alt
            for alt in available
            if need <= frozenset[str]().union(*(fragments[p].covers for p in _placeholders(alt)))
        ]
        if not candidates:
            if need:
                raise RenderError(f"no sentence of slot {slot[0]!r} covers {sorted(need)}")
            continue
        choice = rng.choice(candidates)
        names = _placeholders(choice)
        if not names and any(_placeholders(alt) for alt in slot):
            continue  # a field slot whose fields are all absent
        # «23 июня 2022 г.» or «18 палл.» before a full stop gives «..»: keep one dot
        part = _DOUBLE_DOT.sub(".", choice.format_map({n: fragments[n].text for n in names}))
        parts.append(part[:1].upper() + part[1:] if capitalize else part)
        for name in names:
            placed |= fragments[name].covers
            evidence.extend(fragments[name].fields)
    missing = required - placed
    if missing:
        raise RenderError(f"layout did not place {sorted(missing)}")
    text = "".join(
        part + ("" if i == len(parts) - 1 or part.endswith("\n") else joiner)
        for i, part in enumerate(parts)
    )
    return text, evidence


def _json(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _key(value: Any) -> str:
    return json.dumps(_json(value), sort_keys=True, default=str)


def assemble(text: str, evidence: Sequence[RenderedField], draft: RequestDraft) -> RenderedRequest:
    """Gold answer from the evidence: one value -> the card field; two different values ->
    null plus a conflict; no evidence -> null. `missing_fields` comes from the rule."""
    mentions: dict[FieldName, list[str]] = {}
    values: dict[FieldName, list[Any]] = {}
    for item in evidence:
        if item.text not in text:
            raise RenderError(f"evidence {item.text!r} for {item.field} is not in the text")
        mentions.setdefault(item.field, []).append(item.text)
        known = [_key(value) for value in values.get(item.field, [])]
        if _key(item.gold_value) not in known:
            values.setdefault(item.field, []).append(item.gold_value)
    card: dict[str, Any] = dict.fromkeys(ShipmentCard.model_fields)
    conflicts = []
    for name, found in values.items():
        if len(found) == 1:
            card[name] = found[0]
        else:
            conflicts.append(Conflict(field=name, values=[_json(v) for v in found]))
    shipment = ShipmentCard(**card)
    target = ExtractionTarget(
        card=shipment, missing_fields=compute_missing_fields(shipment), conflicts=conflicts
    )
    weight_in_text = "weight_total" in mentions or "weight_per_piece" in mentions
    variant = VariantInfo(
        weight_unit=draft.weight_unit if weight_in_text else None,
        weight_mode=draft.weight_mode if weight_in_text else "none",
        dropped_fields=draft.dropped_fields,
        hard_cases=draft.hard_cases,
        request_date=draft.request_date,
        date_style=draft.date_style,
        city_lang=draft.city_lang,
        ood_reason=draft.ood_reason,
    )
    return RenderedRequest(text=text, evidence=mentions, target=target, variant=variant)


class LayoutFamily:
    """Base of the template families: `slots` (name -> alternatives) and `orders` (sequences of
    slot names, at least five). Subclasses set the class attributes and register themselves."""

    name: str
    language: Language
    ood_only: bool = False
    joiner: str = " "
    compact: bool = False  # slang: abbreviations, no space before units
    capitalize: bool = True  # sentences start with a capital letter (slang stays lowercase)
    preposition: bool = True  # EN prose: «on March 5, 2022»; key-value forms turn it off
    slots: Mapping[str, Slot]
    orders: tuple[tuple[str, ...], ...]

    def layouts(self) -> tuple[Layout, ...]:
        return tuple(tuple(self.slots[name] for name in order) for order in self.orders)

    def render(self, facts: LoadFacts, draft: RequestDraft, rng: Random) -> RenderedRequest:
        if draft.language != self.language:
            raise RenderError(f"family {self.name} renders {self.language}, not {draft.language}")
        fragments = build_fragments(
            facts, draft, rng, compact=self.compact, preposition=self.preposition
        )
        layouts = self.layouts()
        text, evidence = fill_layout(layouts[rng.randrange(len(layouts))], fragments, rng,
                                     self.joiner, capitalize=self.capitalize)  # fmt: skip
        return assemble(text, evidence, draft)
