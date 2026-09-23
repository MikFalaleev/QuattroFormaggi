"""The SFT dataset generator (plan 6.7, 6.9): LoadFacts -> SFTRecord with gold answers.

Everything is deterministic: loads are sampled with a seeded RNG from facts sorted by load_id,
and every record has its own RNG `Random(f"{seed}:{load_id}:{n}")`, so a record does not
depend on the order of processing. Sets are never iterated while drawing random numbers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from random import Random
from typing import Any, Final, Literal, TypeVar

from pydantic import Field, model_validator

from qf.common import (
    ArtifactRef,
    QFError,
    StrictConfig,
    atomic_write_text,
    read_artifact,
    sha256_file,
    start_run,
    write_artifact,
)
from qf.contracts import (
    SFT_RECORD_SCHEMA_VERSION,
    HardCase,
    Language,
    LoadFacts,
    Message,
    OodReason,
    RenderedRequest,
    RequestDraft,
    SFTRecord,
    TemplateFamily,
    supported_versions,
)
from qf.data.facts import load_facts
from qf.data.registries import HARD_CASES, TEMPLATE_FAMILIES
from qf.data.render import HardCaseNotApplicable
from qf.domain import (
    build_messages,
    check_record,
    get_task,
    great_circle_km,
    serialize_target,
    system_prompt_hash,
)

__all__ = [
    "GenerateConfig",
    "GenerateResult",
    "GeneratedRecord",
    "SamplePlan",
    "generate_dataset",
    "generate_record",
    "generate_records",
    "render_record",
    "sample_loads",
]

GeneratedSplit = Literal["train", "val", "test", "test_ood"]
_SPLITS: Final[tuple[GeneratedSplit, ...]] = ("train", "val", "test", "test_ood")
_MAIN_SPLITS: Final[tuple[GeneratedSplit, ...]] = ("train", "val", "test")
_RUN_KIND = "generate"
T = TypeVar("T")


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
    """`configs/data/generate_v1.yaml`."""

    dataset_name: str = Field(pattern=r"^[a-z0-9_]+$")
    seed: int
    task: str
    schema_version: Literal["card_v1"]
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

    @model_validator(mode="after")
    def _consistent(self) -> GenerateConfig:
        for name, shares in (("language_share", self.language_share), ("mix", self.mix),
                             ("weight_unit_share", self.weight_unit_share)):  # fmt: skip
            if not shares or any(v < 0 for v in shares.values()):
                raise ValueError(f"{name} must have non-negative shares")
            if abs(sum(shares.values()) - 1.0) > 1e-9:
                raise ValueError(f"{name} must sum to 1")
        unknown = sorted(set(self.stratify_by) - set(LoadFacts.model_fields))
        if unknown:
            raise ValueError(f"stratify_by names unknown LoadFacts fields: {unknown}")
        return self


def _check_against_registries(cfg: GenerateConfig) -> None:
    """Names in the config must exist; OOD families are exactly the `ood_only` ones."""
    task = get_task(cfg.task)
    if task.system_prompt_version != cfg.system_prompt_version:
        raise QFError(
            f"task {cfg.task} uses prompt {task.system_prompt_version}, "
            f"config says {cfg.system_prompt_version}"
        )
    if task.target_schema_version != cfg.schema_version:
        raise QFError(f"task {cfg.task} answers in {task.target_schema_version}")
    for name in cfg.families:
        TEMPLATE_FAMILIES.get(name)
    for name in [*cfg.hard, *(key for key in cfg.mix if key not in ("clean", "hard"))]:
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

    splits: dict[GeneratedSplit, list[LoadFacts]]
    ood_reasons: dict[str, OodReason]  # load_id -> reason, only for test_ood
    holdout_routes: list[str]


def _holdout_routes(facts: Sequence[LoadFacts], count: int, rng: Random) -> list[str]:
    """`count` routes spread over route lengths, plus the reverse direction of each (D-055)."""
    routes = sorted({(f.route_id, f.origin.city, f.destination.city) for f in facts})
    if count > len(routes):
        raise QFError(f"cannot hold out {count} of {len(routes)} routes")
    by_length = sorted(routes, key=lambda r: (great_circle_km(r[1], r[2]), r[0]))
    bounds = [round(i * len(by_length) / count) for i in range(count + 1)]
    chosen = [rng.choice(by_length[bounds[i] : bounds[i + 1]]) for i in range(count)]
    pairs = {(origin, destination) for _, origin, destination in chosen}
    return sorted(r[0] for r in routes if (r[1], r[2]) in pairs or (r[2], r[1]) in pairs)


def _stratified(
    items: Sequence[LoadFacts], n: int, keys: Sequence[str], rng: Random
) -> list[LoadFacts]:
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


def _interleave(items: Sequence[T], sizes: Mapping[str, int]) -> dict[str, list[T]]:
    """Deal ordered items to labels in proportion to `sizes`, spreading each label evenly."""
    labels = list(sizes)
    total = sum(sizes.values())
    dealt: dict[str, list[T]] = {label: [] for label in labels}
    for i, item in enumerate(items):
        label = max(labels, key=lambda k: sizes[k] * (i + 1) / total - len(dealt[k]))
        dealt[label].append(item)
    return dealt


def sample_loads(facts: Sequence[LoadFacts], cfg: GenerateConfig) -> SamplePlan:
    """Held-out routes go only to test_ood; the family-OOD reserve and train/val/test never
    share a load (plan 6.9)."""
    rng = Random(f"{cfg.seed}:sample")
    ordered = sorted(facts, key=lambda f: f.load_id)
    holdout = _holdout_routes(ordered, cfg.ood.holdout_route_count, rng)
    held = set(holdout)
    regular = [f for f in ordered if f.route_id not in held]
    on_holdout = [f for f in ordered if f.route_id in held]
    part = cfg.pool.test_ood // 3
    family_reserve = _stratified(regular, cfg.pool.test_ood - 2 * part, cfg.stratify_by, rng)
    reserved = {f.load_id for f in family_reserve}
    rest = [f for f in regular if f.load_id not in reserved]
    sizes: dict[str, int] = {split: getattr(cfg.pool, split) for split in _MAIN_SPLITS}
    main = _interleave(_stratified(rest, sum(sizes.values()), cfg.stratify_by, rng), sizes)
    ood = _interleave(_stratified(on_holdout, 2 * part, cfg.stratify_by, rng),
                      {"route": part, "both": part})  # fmt: skip
    reasons: dict[str, OodReason] = {f.load_id: "family" for f in family_reserve}
    reasons |= {f.load_id: "route" for f in ood["route"]}
    reasons |= {f.load_id: "both" for f in ood["both"]}
    test_ood = sorted([*family_reserve, *ood["route"], *ood["both"]], key=lambda f: f.load_id)
    splits: dict[GeneratedSplit, list[LoadFacts]] = {
        "train": main["train"], "val": main["val"], "test": main["test"], "test_ood": test_ood,
    }  # fmt: skip
    return SamplePlan(splits=splits, ood_reasons=reasons, holdout_routes=holdout)


# --- one record ----------------------------------------------------------------------------


def _base_draft(
    facts: LoadFacts, cfg: GenerateConfig, language: Language, ood: OodReason | None, rng: Random
) -> RequestDraft:
    request_date = facts.pickup_date - timedelta(days=rng.randint(0, 7))
    unit: Literal["kg", "t"] = _weighted(rng, cfg.weight_unit_share)
    return RequestDraft(
        language=language,
        request_date=request_date,
        weight_unit=unit,
        weight_mode="total",
        date_style=rng.choice(("iso", "text")),
        city_lang=language,
        dropped_fields=[],
        conflicts={},
        distractors={},
        hard_cases=[],
        ood_reason=ood,
    )


def _apply_kind(
    kind: str, draft: RequestDraft, facts: LoadFacts, cfg: GenerateConfig, rng: Random
) -> RequestDraft:
    """ "clean" keeps the draft; "hard" applies one case drawn uniformly from `cfg.hard`
    (another one if it does not apply); any other kind is the name of a hard case."""
    if kind == "clean":
        return draft
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
    facts: LoadFacts,
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
    facts: LoadFacts,
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


def generate_records(
    facts: Sequence[LoadFacts], cfg: GenerateConfig, *, source_prefix: str
) -> tuple[dict[GeneratedSplit, list[SFTRecord]], SamplePlan]:
    """All records of the dataset by split, each split sorted by id."""
    _check_against_registries(cfg)
    plan = sample_loads(facts, cfg)
    records: dict[GeneratedSplit, list[SFTRecord]] = {}
    for split in _SPLITS:
        records[split] = sorted(
            (
                generate_record(
                    f,
                    split,
                    n,
                    cfg,
                    ood_reason=plan.ood_reasons.get(f.load_id),
                    source_prefix=source_prefix,
                )  # fmt: skip
                for f in plan.splits[split]
                for n in range(cfg.variants_per_load)
            ),
            key=lambda record: record.id,
        )
    return records, plan


# --- the stage -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateResult:
    refs: dict[str, ArtifactRef]  # split name (and "smoke") -> artifact
    counts: dict[str, int]
    run_dir: Path


def _write_split(
    records: Sequence[SFTRecord], path: Path, *, run_id: str, parent: str, root: Path,
    extra: Mapping[str, Any],
) -> ArtifactRef:  # fmt: skip
    atomic_write_text(path, "".join(record.model_dump_json() + "\n" for record in records))
    return write_artifact(path, "sft_dataset", SFT_RECORD_SCHEMA_VERSION, run_id, [parent],
                          extra=extra, root=root)  # fmt: skip


def _counts(records: Sequence[SFTRecord]) -> dict[str, dict[str, int]]:
    def count(values: Sequence[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    return {
        "family": count([r.template_family for r in records]),
        "language": count([r.language for r in records]),
        "hard_case": count([",".join(r.variant.hard_cases) or "clean" for r in records]),
    }


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
    facts_ref = read_artifact(facts_path, "load_facts", supported_versions("load_facts"),
                              root=root)  # fmt: skip
    records, plan = generate_records(load_facts(facts_ref, root), cfg, source_prefix=source_prefix)
    shared = {"holdout_routes": plan.holdout_routes,
              "holdout_families": sorted(cfg.ood.holdout_families)}  # fmt: skip
    outputs: dict[str, Sequence[SFTRecord]] = {name: items for name, items in records.items()}
    outputs["smoke"] = records["train"][: cfg.pool.smoke]
    refs = {
        name: _write_split(
            items,
            out_dir / f"{name}.jsonl",
            run_id=run.run_id,
            parent=facts_ref.sha256,
            root=root,
            extra={"split": name, "count": len(items), **shared, **_counts(items)},
        )  # fmt: skip
        for name, items in outputs.items()
    }
    counts = {name: len(items) for name, items in outputs.items()}
    run.finish(
        seed=cfg.seed,
        config=cfg.model_dump(mode="json"),
        prompt_hashes={cfg.system_prompt_version: system_prompt_hash(cfg.system_prompt_version)},
        data_hashes={facts_ref.path.as_posix(): facts_ref.sha256,
                     config_path.relative_to(root).as_posix(): sha256_file(config_path),
                     **{ref.path.as_posix(): ref.sha256 for ref in refs.values()}},
        metrics={"counts": counts, "holdout_routes": plan.holdout_routes},
    )  # fmt: skip
    return GenerateResult(refs=refs, counts=counts, run_dir=run.run_dir)
