"""The SFT dataset generator (plan 6.7, 6.9): LoadFacts -> SFTRecord with gold answers.

Everything is deterministic: loads are sampled with a seeded RNG from facts sorted by load_id,
and every record has its own RNG `Random(f"{seed}:{load_id}:{n}")`, so a record does not
depend on the order of processing. Sets are never iterated while drawing random numbers.

The same generator writes card_v1 (`generate_v1.yaml`, facts `load_facts_v1`) and card_v2
(`generate_v2.yaml`, facts `load_facts_v2`, sub-step V3). The card_v2 settings of the config
(`condition_hard`, `length_unit_share`, `profile_floor`) are empty for card_v1, and every
random draw they add happens only when they are set, so card_v1 output does not change.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from random import Random
from typing import Any, Final, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from qf.common import (
    ArtifactRef,
    QFError,
    StrictConfig,
    atomic_write_text,
    read_artifact,
    sha256_file,
    sha256_text,
    start_run,
)
from qf.contracts import (
    LOAD_FACTS_SCHEMA_VERSION,
    LOAD_FACTS_V2_SCHEMA_VERSION,
    AnyLoadFacts,
    HardCase,
    Language,
    LoadFacts,
    LoadFactsV2,
    Message,
    OodReason,
    RenderedRequest,
    RequestDraft,
    SFTRecord,
    TargetSchemaVersion,
    TemplateFamily,
)
from qf.data.facts import load_facts
from qf.data.registries import HARD_CASES, TEMPLATE_FAMILIES
from qf.data.render import HardCaseNotApplicable
from qf.data.review import curate, render_review
from qf.data.sft_io import write_sft_records
from qf.domain import (
    build_messages,
    check_record,
    get_task,
    great_circle_km,
    serialize_target,
    system_prompt_hash,
)

__all__ = [
    "FACTS_FOR_SCHEMA",
    "REVIEW_SAMPLES_FILENAME",
    "GenerateConfig",
    "GenerateResult",
    "GeneratedRecord",
    "SamplePlan",
    "generate_dataset",
    "generate_record",
    "generate_records",
    "generate_rendered",
    "render_record",
    "sample_loads",
]

GeneratedSplit = Literal["train", "val", "test", "test_ood"]
_SPLITS: Final[tuple[GeneratedSplit, ...]] = ("train", "val", "test", "test_ood")
_MAIN_SPLITS: Final[tuple[GeneratedSplit, ...]] = ("train", "val", "test")
_RUN_KIND = "generate"
REVIEW_SAMPLES_FILENAME: Final = "review_samples.md"
T = TypeVar("T")
F = TypeVar("F", bound=LoadFacts | LoadFactsV2)
FACTS_FOR_SCHEMA: Final[dict[str, tuple[str, type[BaseModel]]]] = {
    "card_v1": (LOAD_FACTS_SCHEMA_VERSION, LoadFacts),
    "card_v2": (LOAD_FACTS_V2_SCHEMA_VERSION, LoadFactsV2),
}
"""The facts version and model an answer schema is generated from (D-094)."""


class PoolConfig(StrictConfig):
    train: int = Field(ge=1)
    val: int = Field(ge=1)
    test: int = Field(ge=1)
    test_ood: int = Field(ge=3)  # three equal parts: route / family / both
    smoke: int = Field(ge=1)


class OodConfig(StrictConfig):
    holdout_route_count: int = Field(ge=1)
    holdout_families: list[str]


class GenerateConfig(StrictConfig):
    """`configs/data/generate_v1.yaml`, `configs/data/generate_v2.yaml`."""

    dataset_name: str = Field(pattern=r"^[a-z0-9_]+$")
    seed: int
    task: str
    schema_version: TargetSchemaVersion
    system_prompt_version: str
    language_share: dict[Language, float]
    variants_per_load: int = Field(ge=1)
    pool: PoolConfig
    mix: dict[str, float]
    hard: list[str]
    weight_unit_share: dict[Literal["kg", "t"], float]  # no pounds (D-049)
    families: list[str]
    ood: OodConfig
    stratify_by: list[str]
    # card_v2 only (sub-step V3, D-095)
    condition_hard: dict[str, float] = Field(default_factory=dict)  # case -> weight ("conditions")
    length_unit_share: dict[Literal["m", "cm"], float] | None = None
    profile_floor: float | None = Field(default=None, gt=0, lt=1)  # min share of each profile
    # minimum records of a hard case per split; clean records that it applies to are
    # re-rendered with it until the minimum is met (rare cases, e.g. lowbed-only ones)
    hard_case_floor: dict[GeneratedSplit, dict[str, int]] = Field(default_factory=dict)
    review_samples: int = Field(default=0, ge=0)  # curated samples for a person (0: none)

    @model_validator(mode="after")
    def _consistent(self) -> GenerateConfig:
        shares_of: list[tuple[str, Mapping[Any, float]]] = [
            ("language_share", self.language_share), ("mix", self.mix),
            ("weight_unit_share", self.weight_unit_share),
        ]  # fmt: skip
        if self.length_unit_share is not None:
            shares_of.append(("length_unit_share", self.length_unit_share))
        for name, shares in shares_of:
            if not shares or any(v < 0 for v in shares.values()):
                raise ValueError(f"{name} must have non-negative shares")
            if abs(sum(shares.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} must sum to 1")
        facts_model = FACTS_FOR_SCHEMA[self.schema_version][1]
        unknown = sorted(set(self.stratify_by) - set(facts_model.model_fields))
        if unknown:
            raise ValueError(f"stratify_by names unknown {facts_model.__name__} fields: {unknown}")
        if any(weight <= 0 for weight in self.condition_hard.values()):
            raise ValueError("condition_hard weights must be positive")
        if ("conditions" in self.mix) != bool(self.condition_hard):
            raise ValueError("mix key 'conditions' and condition_hard go together")
        conditions_facts = "special_conditions" in facts_model.model_fields
        if self.length_unit_share is None and conditions_facts:
            raise ValueError(f"{self.schema_version} needs length_unit_share")
        if not conditions_facts and (self.length_unit_share is not None or self.condition_hard
                                     or self.profile_floor is not None):  # fmt: skip
            raise ValueError(f"{self.schema_version} has no special conditions to configure")
        if self.profile_floor is not None and "profile" not in facts_model.model_fields:
            raise ValueError("profile_floor needs facts with a profile")
        floors = [n for split in self.hard_case_floor.values() for n in split.values()]
        if any(n < 1 for n in floors):
            raise ValueError("hard_case_floor minimums must be positive")
        return self


def _check_against_registries(cfg: GenerateConfig) -> None:
    """Names in the config must exist; OOD families are exactly the `ood_only` ones."""
    task = get_task(cfg.task)
    prompt = task.prompt_for(cfg.schema_version)
    if prompt != cfg.system_prompt_version:
        raise QFError(
            f"task {cfg.task} uses prompt {prompt} for {cfg.schema_version}, "
            f"config says {cfg.system_prompt_version}"
        )
    for name in cfg.families:
        TEMPLATE_FAMILIES.get(name)
    groups = ("clean", "hard", "conditions")
    floors = [name for split in cfg.hard_case_floor.values() for name in split]
    for name in [*cfg.hard, *cfg.condition_hard, *floors,
                 *(key for key in cfg.mix if key not in groups)]:  # fmt: skip
        HARD_CASES.get(name)
    ood_only = sorted(name for name in cfg.families if TEMPLATE_FAMILIES.get(name)().ood_only)
    if ood_only != sorted(cfg.ood.holdout_families):
        raise QFError(
            f"ood.holdout_families {sorted(cfg.ood.holdout_families)} must equal the "
            f"ood_only families of the config {ood_only}"
        )
    for language in sorted(cfg.language_share):
        for ood in (False, True):
            if not _family_names(cfg, language, ood):
                raise QFError(f"no {'OOD' if ood else 'regular'} family for language {language}")


def _family_names(cfg: GenerateConfig, language: Language, ood_only: bool) -> list[str]:
    names = []
    for name in sorted(cfg.families):
        family = TEMPLATE_FAMILIES.get(name)()
        if family.language == language and family.ood_only == ood_only:
            names.append(name)
    return names


def _weighted(rng: Random, shares: Mapping[T, float]) -> T:
    keys = sorted(shares, key=str)
    return rng.choices(keys, weights=[shares[key] for key in keys])[0]


# --- sampling ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePlan:
    """Which load goes to which split, and why a test_ood load is out of distribution."""

    splits: dict[GeneratedSplit, list[AnyLoadFacts]]
    ood_reasons: dict[str, OodReason]  # load_id -> reason, only for test_ood
    holdout_routes: list[str]


def _holdout_routes(facts: Sequence[AnyLoadFacts], count: int, rng: Random) -> list[str]:
    """`count` routes spread over route lengths, plus the reverse direction of each (D-055)."""
    routes = sorted({(f.route_id, f.origin.city, f.destination.city) for f in facts})
    if count > len(routes):
        raise QFError(f"cannot hold out {count} of {len(routes)} routes")
    by_length = sorted(routes, key=lambda r: (great_circle_km(r[1], r[2]), r[0]))
    bounds = [round(i * len(by_length) / count) for i in range(count + 1)]
    chosen = [rng.choice(by_length[bounds[i] : bounds[i + 1]]) for i in range(count)]
    pairs = {(origin, destination) for _, origin, destination in chosen}
    return sorted(r[0] for r in routes if (r[1], r[2]) in pairs or (r[2], r[1]) in pairs)


def _stratified(items: Sequence[F], n: int, keys: Sequence[str], rng: Random) -> list[F]:
    """Systematic sample of `n` items ordered by the strata keys (random order inside strata),
    so every stratum is represented in proportion to its size."""
    if n > len(items):
        raise QFError(f"need {n} loads, only {len(items)} are available")
    ordered = sorted(items, key=lambda f: f.load_id)
    rng.shuffle(ordered)
    ordered.sort(key=lambda f: tuple(str(getattr(f, key)) for key in keys))
    step = len(ordered) / n
    offset = rng.random() * step
    return [ordered[min(len(ordered) - 1, math.floor(offset + i * step))] for i in range(n)]


def _floored(items: Sequence[F], n: int, keys: Sequence[str], rng: Random,
             floor: float) -> list[F]:  # fmt: skip
    """`_stratified` inside each table profile. The share of every profile is raised to at
    least `floor` and the shares are then normalized to 1, so rare equipment and conditions
    reach every split (a profile never gets more loads than it has). Profiles come in name
    order; the result is ordered by profile, then by the strata keys."""
    groups: dict[str, list[F]] = defaultdict(list)
    for fact in sorted(items, key=lambda f: f.load_id):
        groups[getattr(fact, "profile")].append(fact)  # noqa: B009  (only v2 facts have it)
    names = sorted(groups)
    wanted = {p: max(len(groups[p]) / len(items), floor) for p in names}
    total = sum(wanted.values())
    quotas = {p: n * wanted[p] / total for p in names}
    counts = {p: min(len(groups[p]), math.floor(quotas[p])) for p in names}
    by_remainder = sorted(names, key=lambda p: (-(quotas[p] - math.floor(quotas[p])), p))
    while sum(counts.values()) < n:
        grown = False
        for p in by_remainder:
            if sum(counts.values()) < n and counts[p] < len(groups[p]):
                counts[p] += 1
                grown = True
        if not grown:
            raise QFError(f"need {n} loads, only {len(items)} are available")
    return [f for p in names if counts[p] for f in _stratified(groups[p], counts[p], keys, rng)]


def _sample(items: Sequence[F], n: int, cfg: GenerateConfig, rng: Random) -> list[F]:
    if cfg.profile_floor is None:
        return _stratified(items, n, cfg.stratify_by, rng)
    keys = [key for key in cfg.stratify_by if key != "profile"]
    return _floored(items, n, keys, rng, cfg.profile_floor)


def _interleave(items: Sequence[T], sizes: Mapping[str, int]) -> dict[str, list[T]]:
    """Deal ordered items to labels in proportion to `sizes`, spreading each label evenly."""
    labels = list(sizes)
    total = sum(sizes.values())
    dealt: dict[str, list[T]] = {label: [] for label in labels}
    for i, item in enumerate(items):
        label = max(labels, key=lambda k: sizes[k] * (i + 1) / total - len(dealt[k]))
        dealt[label].append(item)
    return dealt


def sample_loads(facts: Sequence[AnyLoadFacts], cfg: GenerateConfig) -> SamplePlan:
    """Held-out routes go only to test_ood; the family-OOD reserve and train/val/test never
    share a load (plan 6.9)."""
    rng = Random(f"{cfg.seed}:sample")
    ordered = sorted(facts, key=lambda f: f.load_id)
    holdout = _holdout_routes(ordered, cfg.ood.holdout_route_count, rng)
    held = set(holdout)
    regular = [f for f in ordered if f.route_id not in held]
    on_holdout = [f for f in ordered if f.route_id in held]
    part = cfg.pool.test_ood // 3
    family_reserve = _sample(regular, cfg.pool.test_ood - 2 * part, cfg, rng)
    reserved = {f.load_id for f in family_reserve}
    rest = [f for f in regular if f.load_id not in reserved]
    sizes: dict[str, int] = {split: getattr(cfg.pool, split) for split in _MAIN_SPLITS}
    main = _interleave(_sample(rest, sum(sizes.values()), cfg, rng), sizes)
    ood = _interleave(_sample(on_holdout, 2 * part, cfg, rng),
                      {"route": part, "both": part})  # fmt: skip
    reasons: dict[str, OodReason] = {f.load_id: "family" for f in family_reserve}
    reasons |= {f.load_id: "route" for f in ood["route"]}
    reasons |= {f.load_id: "both" for f in ood["both"]}
    test_ood = sorted([*family_reserve, *ood["route"], *ood["both"]], key=lambda f: f.load_id)
    splits: dict[GeneratedSplit, list[AnyLoadFacts]] = {
        "train": main["train"], "val": main["val"], "test": main["test"], "test_ood": test_ood,
    }  # fmt: skip
    return SamplePlan(splits=splits, ood_reasons=reasons, holdout_routes=holdout)


# --- one record ----------------------------------------------------------------------------


def _base_draft(
    facts: AnyLoadFacts, cfg: GenerateConfig, language: Language, ood: OodReason | None,
    rng: Random,
) -> RequestDraft:  # fmt: skip
    request_date = facts.pickup_date - timedelta(days=rng.randint(0, 7))
    unit: Literal["kg", "t"] = _weighted(rng, cfg.weight_unit_share)
    styles: tuple[Literal["iso", "text"], ...] = ("iso", "text")
    date_style = rng.choice(styles)
    length_unit: Literal["m", "cm"] = "m"
    if cfg.length_unit_share is not None:  # card_v2 only: after the card_v1 draws
        length_unit = _weighted(rng, cfg.length_unit_share)
    return RequestDraft(
        language=language,
        request_date=request_date,
        weight_unit=unit,
        weight_mode="total",
        date_style=date_style,
        city_lang=language,
        dropped_fields=[],
        conflicts={},
        distractors={},
        hard_cases=[],
        ood_reason=ood,
        schema_version=cfg.schema_version,
        length_unit=length_unit,
    )


def _apply_condition_case(
    draft: RequestDraft, facts: AnyLoadFacts, cfg: GenerateConfig, rng: Random
) -> RequestDraft:
    """One case of `cfg.condition_hard` drawn by weight among those that apply to the load;
    a load without applicable cases stays clean."""
    cases = [(name, HARD_CASES.get(name)()) for name in sorted(cfg.condition_hard)]
    usable = [(name, case) for name, case in cases if case.applicable(facts)]
    if not usable:
        return draft
    weights = [cfg.condition_hard[name] for name, _ in usable]
    case: HardCase = rng.choices([case for _, case in usable], weights=weights)[0]
    try:
        return case.apply(draft, facts, rng)
    except HardCaseNotApplicable:
        return draft


def _apply_kind(
    kind: str, draft: RequestDraft, facts: AnyLoadFacts, cfg: GenerateConfig, rng: Random
) -> RequestDraft:
    """ "clean" keeps the draft; "hard" applies one case drawn uniformly from `cfg.hard`
    (another one if it does not apply); "conditions" one case of `cfg.condition_hard` drawn by
    weight among the applicable ones (card_v2); any other kind is the name of a hard case."""
    if kind == "clean":
        return draft
    if kind == "conditions":
        return _apply_condition_case(draft, facts, cfg, rng)
    names = list(cfg.hard) if kind == "hard" else [kind]
    rng.shuffle(names)
    for name in names:
        case: HardCase = HARD_CASES.get(name)()
        if not case.applicable(facts):
            continue
        try:
            return case.apply(draft, facts, rng)
        except HardCaseNotApplicable:
            continue
    return draft


@dataclass(frozen=True)
class GeneratedRecord:
    """A record with the draft and rendering it came from (tests check them against it)."""

    draft: RequestDraft
    rendered: RenderedRequest
    record: SFTRecord


def render_record(
    facts: AnyLoadFacts,
    split: GeneratedSplit,
    n: int,
    cfg: GenerateConfig,
    *,
    ood_reason: OodReason | None,
    source_prefix: str,
    family: str | None = None,
    kind: str | None = None,
) -> GeneratedRecord:
    """One record; `family` and `kind` ("clean", "hard" or a hard case name) are drawn from the
    config unless given. The record is checked with `check_record` before it is returned."""
    rng = Random(f"{cfg.seed}:{facts.load_id}:{n}")
    if family is None:
        language: Language = _weighted(rng, cfg.language_share)
        ood_family = ood_reason in ("family", "both")
        family = rng.choice(_family_names(cfg, language, ood_only=ood_family))
    template: TemplateFamily = TEMPLATE_FAMILIES.get(family)()
    language = template.language
    kind = kind if kind is not None else _weighted(rng, cfg.mix)
    draft = _apply_kind(kind, _base_draft(facts, cfg, language, ood_reason, rng), facts, cfg, rng)
    rendered = template.render(facts, draft, rng)
    messages = [
        *build_messages(rendered.text, draft.request_date, language, cfg.system_prompt_version),
        Message(role="assistant", content=serialize_target(rendered.target)),
    ]
    record = SFTRecord(
        id=f"qf-{split}-{facts.load_id}-{n}",
        task=cfg.task,
        group_id=f"load:{facts.load_id}",
        source_id=f"{source_prefix}:{facts.load_id}",
        language=language,
        reviewed=False,
        synthetic=True,
        template_family=template.name,
        variant=rendered.variant,
        schema_version=cfg.schema_version,
        split=split,
        messages=messages,
    )
    issues = check_record(record)
    if issues:
        raise QFError(f"generator produced an invalid record {record.id}: {issues}")
    return GeneratedRecord(draft=draft, rendered=rendered, record=record)


def generate_record(
    facts: AnyLoadFacts,
    split: GeneratedSplit,
    n: int,
    cfg: GenerateConfig,
    *,
    ood_reason: OodReason | None,
    source_prefix: str,
) -> SFTRecord:
    """One SFT record with family and kind drawn from the config."""
    return render_record(
        facts, split, n, cfg, ood_reason=ood_reason, source_prefix=source_prefix
    ).record


def _top_up(
    split: GeneratedSplit, items: list[GeneratedRecord], plan: SamplePlan, cfg: GenerateConfig,
    source_prefix: str,
) -> list[GeneratedRecord]:  # fmt: skip
    """Meet `cfg.hard_case_floor[split]`: clean records whose load the case applies to are
    re-rendered with it, in a seeded order of loads (independent of processing order)."""
    floor = cfg.hard_case_floor.get(split, {})
    if not floor:
        return items
    by_id = {g.record.id: g for g in items}
    loads = sorted(plan.splits[split], key=lambda f: sha256_text(f"{cfg.seed}:{f.load_id}"))
    for name in sorted(floor):
        case: HardCase = HARD_CASES.get(name)()
        have = sum(1 for g in by_id.values() if name in g.record.variant.hard_cases)
        for facts in loads:
            if have >= floor[name]:
                break
            record_id = f"qf-{split}-{facts.load_id}-0"
            current = by_id[record_id].record.variant
            if current.hard_cases or current.dropped_fields or not case.applicable(facts):
                continue
            redone = render_record(
                facts,
                split,
                0,
                cfg,
                source_prefix=source_prefix,
                ood_reason=plan.ood_reasons.get(facts.load_id),
                kind=name,
            )
            if name in redone.record.variant.hard_cases:
                by_id[record_id] = redone
                have += 1
        if have < floor[name]:
            raise QFError(f"{split}: only {have} records can take {name}, "
                          f"hard_case_floor asks for {floor[name]}")  # fmt: skip
    return list(by_id.values())


def generate_rendered(
    facts: Sequence[AnyLoadFacts], cfg: GenerateConfig, *, source_prefix: str
) -> tuple[dict[GeneratedSplit, list[GeneratedRecord]], SamplePlan]:
    """All records of the dataset with their drafts, by split, each split sorted by id."""
    _check_against_registries(cfg)
    plan = sample_loads(facts, cfg)
    rendered: dict[GeneratedSplit, list[GeneratedRecord]] = {}
    for split in _SPLITS:
        items = [
            render_record(
                f,
                split,
                n,
                cfg,
                ood_reason=plan.ood_reasons.get(f.load_id),
                source_prefix=source_prefix,
            )  # fmt: skip
            for f in plan.splits[split]
            for n in range(cfg.variants_per_load)
        ]
        items = _top_up(split, items, plan, cfg, source_prefix)
        rendered[split] = sorted(items, key=lambda g: g.record.id)
    return rendered, plan


def generate_records(
    facts: Sequence[AnyLoadFacts], cfg: GenerateConfig, *, source_prefix: str
) -> tuple[dict[GeneratedSplit, list[SFTRecord]], SamplePlan]:
    """All records of the dataset by split, each split sorted by id."""
    rendered, plan = generate_rendered(facts, cfg, source_prefix=source_prefix)
    return {split: [g.record for g in items] for split, items in rendered.items()}, plan


# --- the stage -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateResult:
    refs: dict[str, ArtifactRef]  # split name (and "smoke") -> artifact
    counts: dict[str, int]
    run_dir: Path


def _counts(records: Sequence[SFTRecord]) -> dict[str, dict[str, int]]:
    def count(values: Sequence[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    return {
        "family": count([r.template_family for r in records]),
        "language": count([r.language for r in records]),
        "hard_case": count([",".join(r.variant.hard_cases) or "clean" for r in records]),
    }


def _relative(path: Path, root: Path) -> str:
    """The path relative to the project root when it is inside it (a relative `--config`
    counts from the working directory), else absolute."""
    absolute, base = path.resolve(), root.resolve()
    return absolute.relative_to(base).as_posix() if absolute.is_relative_to(base) else str(absolute)


def generate_dataset(
    facts_path: Path,
    cfg: GenerateConfig,
    *,
    root: Path,
    out_dir: Path,
    source_prefix: str,
    config_path: Path,
) -> GenerateResult:
    """The generate stage: `load_facts` artifact -> one `sft_dataset@sft_record_v1` artifact per
    split plus `smoke.jsonl` (the first train records by id) and a run manifest."""
    run = start_run(_RUN_KIND, root)
    facts_version, facts_model = FACTS_FOR_SCHEMA[cfg.schema_version]
    facts_ref = read_artifact(facts_path, "load_facts", {facts_version}, root=root)
    facts = load_facts(facts_ref, root, model=facts_model)
    records, plan = generate_records(facts, cfg, source_prefix=source_prefix)
    shared = {"holdout_routes": plan.holdout_routes,
              "holdout_families": sorted(cfg.ood.holdout_families)}  # fmt: skip
    outputs: dict[str, Sequence[SFTRecord]] = {name: items for name, items in records.items()}
    outputs["smoke"] = records["train"][: cfg.pool.smoke]
    refs = {
        name: write_sft_records(
            items,
            out_dir / f"{name}.jsonl",
            run_id=run.run_id,
            parents=[facts_ref.sha256],
            root=root,
            extra={"split": name, "count": len(items), **shared, **_counts(items)},
        )  # fmt: skip
        for name, items in outputs.items()
    }
    counts = {name: len(items) for name, items in outputs.items()}
    if cfg.review_samples:
        everything = [record for items in records.values() for record in items]
        chosen = curate(everything, seed=cfg.seed, limit=cfg.review_samples)
        atomic_write_text(out_dir / REVIEW_SAMPLES_FILENAME,
                          render_review(dict(records), chosen, cfg.dataset_name))  # fmt: skip
    run.finish(
        seed=cfg.seed,
        config=cfg.model_dump(mode="json"),
        prompt_hashes={cfg.system_prompt_version: system_prompt_hash(cfg.system_prompt_version)},
        data_hashes={facts_ref.path.as_posix(): facts_ref.sha256,
                     _relative(config_path, root): sha256_file(config_path),
                     **{ref.path.as_posix(): ref.sha256 for ref in refs.values()}},
        metrics={"counts": counts, "holdout_routes": plan.holdout_routes},
    )  # fmt: skip
    return GenerateResult(refs=refs, counts=counts, run_dir=run.run_dir)
