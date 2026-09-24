"""Layouts, the gold-answer assembler and the base class of template families (plan 6.3, 6.5).

A family only composes fragments: it lists slots (sentences with alternatives) and orders of
slots. `fill_layout` places every fragment exactly once and `assemble` builds the gold answer
from the evidence that ended up in the text. A layout that loses a field is an error, never a
silently wrong gold answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from random import Random
from typing import Any, Final

from pydantic import BaseModel

from qf.common import QFError
from qf.contracts import (
    CARD_V2_CONDITION_KINDS,
    AnyFieldName,
    AnyLoadFacts,
    Conflict,
    ConflictV2,
    ExtractionTarget,
    ExtractionTargetV2,
    Language,
    LoadFacts,
    LoadFactsV2,
    RenderedField,
    RenderedRequest,
    RequestDraft,
    ShipmentCard,
    ShipmentCardV2,
    SpecialCondition,
    VariantInfo,
)
from qf.data.render.conditions import ConditionStyle, condition_fragments
from qf.data.render.fields import Fragment, build_fragments
from qf.data.render.vocabulary import RU_FEMININE_EQUIPMENT, VOCABULARIES
from qf.domain import compute_missing_fields, compute_missing_fields_v2

__all__ = [
    "ASSEMBLERS",
    "HardCaseNotApplicable",
    "Layout",
    "LayoutFamily",
    "RenderError",
    "Slot",
    "assemble",
    "agree",
    "assemble_v2",
    "fill_layout",
]

Slot = tuple[str, ...]
"""Alternative sentences; `{placeholder}` names a fragment."""
Layout = tuple[Slot, ...]

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_RU_FEMININE_FRAMES: Final[tuple[tuple[str, str], ...]] = (
    ("Нужен {equipment}", "Нужна {equipment}"), ("нужен {equipment}", "нужна {equipment}"),
    ("{equipment} нужен", "{equipment} нужна"), ("ищу {equipment}", "нужна {equipment}"),
)  # fmt: skip
_EN_VOWEL_FRAMES: Final[tuple[tuple[str, str], ...]] = (("a {equipment}", "an {equipment}"),)
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


def _collect(
    text: str, evidence: Sequence[RenderedField]
) -> tuple[dict[AnyFieldName, list[str]], dict[AnyFieldName, list[Any]]]:
    """Mentions and distinct gold values of every field, in text order."""
    mentions: dict[AnyFieldName, list[str]] = {}
    values: dict[AnyFieldName, list[Any]] = {}
    for item in evidence:
        if item.text not in text:
            raise RenderError(f"evidence {item.text!r} for {item.field} is not in the text")
        mentions.setdefault(item.field, []).append(item.text)
        known = [_key(value) for value in values.get(item.field, [])]
        if _key(item.gold_value) not in known:
            values.setdefault(item.field, []).append(item.gold_value)
    return mentions, values


def _variant(mentions: Mapping[AnyFieldName, list[str]], draft: RequestDraft) -> VariantInfo:
    weight_in_text = "weight_total" in mentions or "weight_per_piece" in mentions
    return VariantInfo(
        weight_unit=draft.weight_unit if weight_in_text else None,
        weight_mode=draft.weight_mode if weight_in_text else "none",
        dropped_fields=draft.dropped_fields,
        hard_cases=draft.hard_cases,
        request_date=draft.request_date,
        date_style=draft.date_style,
        city_lang=draft.city_lang,
        ood_reason=draft.ood_reason,
    )


def assemble(text: str, evidence: Sequence[RenderedField], draft: RequestDraft) -> RenderedRequest:
    """Gold answer from the evidence: one value -> the card field; two different values ->
    null plus a conflict; no evidence -> null. `missing_fields` comes from the rule."""
    mentions, values = _collect(text, evidence)
    card: dict[str, Any] = dict.fromkeys(ShipmentCard.model_fields)
    conflicts = []
    for name, found in values.items():
        if len(found) == 1:
            card[name] = found[0]
        else:
            conflicts.append(Conflict(field=name, values=[_json(v) for v in found]))  # type: ignore[arg-type]
    shipment = ShipmentCard(**card)
    target = ExtractionTarget(
        card=shipment, missing_fields=compute_missing_fields(shipment), conflicts=conflicts
    )
    return RenderedRequest(text=text, evidence=mentions, target=target,
                           variant=_variant(mentions, draft))  # fmt: skip


def assemble_v2(
    text: str, evidence: Sequence[RenderedField], draft: RequestDraft
) -> RenderedRequest:
    """`assemble` for card_v2: the conditions are the gold values of the `special_conditions`
    evidence, one per kind, in canonical order; no evidence -> no conditions (`[]`)."""
    mentions, values = _collect(text, evidence)
    conditions: list[SpecialCondition] = values.pop("special_conditions", [])
    kinds = [condition.kind for condition in conditions]
    if len(set(kinds)) != len(kinds):
        raise RenderError(f"a condition kind is written twice: {kinds}")
    card: dict[str, Any] = dict.fromkeys(ShipmentCardV2.model_fields)
    card["special_conditions"] = sorted(
        conditions, key=lambda c: CARD_V2_CONDITION_KINDS.index(c.kind)
    )
    conflicts = []
    for name, found in values.items():
        if len(found) == 1:
            card[name] = found[0]
        else:
            conflicts.append(ConflictV2(field=name, values=[_json(v) for v in found]))  # type: ignore[arg-type]
    shipment = ShipmentCardV2(**card)
    target = ExtractionTargetV2(
        card=shipment, missing_fields=compute_missing_fields_v2(shipment), conflicts=conflicts
    )
    return RenderedRequest(text=text, evidence=mentions, target=target,
                           variant=_variant(mentions, draft))  # fmt: skip


Assembler = Callable[[str, Sequence[RenderedField], RequestDraft], RenderedRequest]
ASSEMBLERS: Final[dict[str, Assembler]] = {"card_v1": assemble, "card_v2": assemble_v2}
"""The gold-answer assembler of each answer schema (D-094)."""


def _no_conditions(
    facts: AnyLoadFacts, draft: RequestDraft, rng: Random, style: ConditionStyle
) -> dict[str, Fragment]:
    return {}


def _conditions_v2(
    facts: AnyLoadFacts, draft: RequestDraft, rng: Random, style: ConditionStyle
) -> dict[str, Fragment]:
    if not isinstance(facts, LoadFactsV2):
        raise RenderError(f"a card_v2 draft needs load_facts_v2, got {type(facts).__name__}")
    return condition_fragments(facts, draft, rng, style)


_CONDITION_FRAGMENTS: Final[
    dict[str, Callable[[AnyLoadFacts, RequestDraft, Random, ConditionStyle], dict[str, Fragment]]]
] = {"card_v1": _no_conditions, "card_v2": _conditions_v2}
_FACTS_TYPES: Final[dict[str, type]] = {"card_v1": LoadFacts, "card_v2": LoadFactsV2}


def agree(layout: Layout, fragments: Mapping[str, Fragment], language: Language) -> Layout:
    """The layout with its equipment sentences agreed with the chosen equipment word: the
    templates say «Нужен {equipment}», «ищу {equipment}» and «a {equipment}», so a feminine word
    gets «Нужна» («ищу фура» becomes «нужна фура») and a word starting with a vowel gets «an».
    The words themselves (the evidence) never change and no random number is drawn."""
    equipment = fragments.get("equipment")
    if equipment is None:
        return layout
    if language == "ru" and equipment.text in RU_FEMININE_EQUIPMENT:
        frames = _RU_FEMININE_FRAMES
    elif language == "en" and equipment.text[:1].lower() in "aeio":
        frames = _EN_VOWEL_FRAMES
    else:
        return layout

    def fix(sentence: str) -> str:
        for old, new in frames:
            sentence = sentence.replace(old, new)
        return sentence

    return tuple(tuple(fix(sentence) for sentence in slot) for slot in layout)


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
    condition_style: ConditionStyle = "prose"  # card_v2 conditions: prose, list or slang
    condition_anchor: str = "equipment"  # conditions follow this slot unless scattered
    slots: Mapping[str, Slot]
    orders: tuple[tuple[str, ...], ...]

    def layouts(self) -> tuple[Layout, ...]:
        return tuple(tuple(self.slots[name] for name in order) for order in self.orders)

    def _layout(self, order: Sequence[str], conditions: Sequence[str], draft: RequestDraft,
                rng: Random) -> Layout:  # fmt: skip
        """The order's slots with one slot per condition fragment: all after the anchor slot,
        or (`conditions_scattered`) each after a different, randomly chosen slot. Without
        conditions (always for card_v1) no random number is drawn."""
        names: list[str] = list(order)
        if conditions:
            placed = list(conditions)
            if len(placed) > 1:
                rng.shuffle(placed)
            if draft.conditions_scattered:
                spots = rng.sample(range(1, len(names)), min(len(placed), len(names) - 1))
                for spot, name in sorted(zip(spots, placed, strict=False), reverse=True):
                    names.insert(spot, name)
                names += placed[len(spots) :]  # more conditions than places: the rest at the end
            else:
                anchor = names.index(self.condition_anchor) + 1
                names[anchor:anchor] = placed
        return tuple(self.slots.get(name, (f"{{{name}}}",)) for name in names)

    def render(self, facts: AnyLoadFacts, draft: RequestDraft, rng: Random) -> RenderedRequest:
        if draft.language != self.language:
            raise RenderError(f"family {self.name} renders {self.language}, not {draft.language}")
        if not isinstance(facts, _FACTS_TYPES[draft.schema_version]):
            raise RenderError(f"a {draft.schema_version} draft cannot render "
                              f"{type(facts).__name__}")  # fmt: skip
        fragments = build_fragments(
            facts, draft, rng, compact=self.compact, preposition=self.preposition,
            vocabulary=VOCABULARIES[draft.schema_version],
        )  # fmt: skip
        conditions = _CONDITION_FRAGMENTS[draft.schema_version](facts, draft, rng,
                                                                 self.condition_style)  # fmt: skip
        fragments |= conditions
        order = self.orders[rng.randrange(len(self.orders))]
        layout = agree(self._layout(order, sorted(conditions), draft, rng), fragments,
                       self.language)  # fmt: skip
        text, evidence = fill_layout(layout, fragments, rng, self.joiner,
                                     capitalize=self.capitalize)  # fmt: skip
        return ASSEMBLERS[draft.schema_version](text, evidence, draft)
