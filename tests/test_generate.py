from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from random import Random

import pytest
import yaml

from qf.cli import main
from qf.common import QFError, load_artifact_manifest, read_artifact, read_manifest, write_artifact
from qf.contracts import LoadFacts, RequestDraft, SFTRecord, supported_versions
from qf.data import (
    HARD_CASES,
    TEMPLATE_FAMILIES,
    GenerateConfig,
    generate_records,
    render_record,
    sample_loads,
)
from qf.data.render import HardCaseNotApplicable
from qf.domain import check_record, system_prompt_hash
from tests.conftest import REPO_ROOT
from tests.generation import (
    FIXTURE,
    INVARIANTS,
    REAL_RAW,
    SOURCE_PREFIX,
    facts_of,
    money_of,
    repo_config,
    small_config,
    synthetic_facts,
    violations,
)

SYNTHETIC_POOL = {"train": 120, "val": 20, "test": 30, "test_ood": 30, "smoke": 10}


def synthetic_config(**changes: object) -> GenerateConfig:
    ood = {"holdout_route_count": 2, "holdout_families": ["T7", "T8"]}
    return repo_config(**{"pool": SYNTHETIC_POOL, "ood": ood, **changes})


# --- sampling and the dataset-level invariants E9, E10 ------------------------------------


def test_sampling_separates_loads_and_holds_out_both_directions() -> None:
    facts = synthetic_facts()
    plan = sample_loads(facts, synthetic_config())
    by_id = {f.load_id: f for f in facts}
    ids = [f.load_id for split in plan.splits.values() for f in split]
    assert len(ids) == len(set(ids)) == 200
    assert {s: len(v) for s, v in plan.splits.items()} == {
        "train": 120, "val": 20, "test": 30, "test_ood": 30,
    }  # fmt: skip
    held = set(plan.holdout_routes)
    pairs = {(by_id[i].origin.city, by_id[i].destination.city) for i in ids
             if by_id[i].route_id in held}  # fmt: skip
    assert all((b, a) in pairs for a, b in pairs)  # both directions of a held-out pair
    for split in ("train", "val", "test"):
        assert all(f.route_id not in held for f in plan.splits[split])
    assert Counter(plan.ood_reasons.values()) == {"route": 10, "both": 10, "family": 10}
    for f in plan.splits["test_ood"]:
        on_holdout = f.route_id in held
        assert on_holdout == (plan.ood_reasons[f.load_id] in ("route", "both"))


def test_sampling_is_stratified() -> None:
    plan = sample_loads(synthetic_facts(), synthetic_config())
    reefers = Counter(f.equipment_type for f in plan.splits["train"])
    assert 30 <= reefers["reefer"] <= 50  # a third of the synthetic loads are reefers


def test_e9_e10_ood_families_and_routes_only_in_test_ood() -> None:
    facts = synthetic_facts()
    records, plan = generate_records(facts, synthetic_config(), source_prefix=SOURCE_PREFIX)
    by_id = {f.load_id: f for f in facts}
    held = set(plan.holdout_routes)
    for split, items in records.items():
        for record in items:
            ood_family = record.template_family in ("T7", "T8")
            on_holdout = by_id[record.group_id.removeprefix("load:")].route_id in held
            reason = record.variant.ood_reason
            if split != "test_ood":
                assert not ood_family and not on_holdout and reason is None, record.id
            else:
                assert ood_family == (reason in ("family", "both")), record.id
                assert on_holdout == (reason in ("route", "both")), record.id
            assert check_record(record) == []


def test_mix_and_languages_follow_the_config() -> None:
    records, _ = generate_records(
        synthetic_facts(), synthetic_config(), source_prefix=SOURCE_PREFIX
    )
    train = records["train"]
    kinds = Counter(r.variant.hard_cases[0] if r.variant.hard_cases else "clean" for r in train)
    assert 45 <= kinds["clean"] <= 85  # 55 % of 120
    assert 15 <= kinds["dropped_fields"] <= 45  # 25 %
    assert len(set(kinds) - {"clean", "dropped_fields"}) >= 4  # hard cases are mixed
    assert 55 <= Counter(r.language for r in train)["ru"] <= 90  # 60 %
    assert {r.variant.weight_unit for r in train} <= {"kg", "t", None}  # no pounds (D-049)


def test_config_is_checked_against_registries() -> None:
    facts = synthetic_facts()
    with pytest.raises(QFError, match="must equal the ood_only families"):
        generate_records(facts, synthetic_config(ood={"holdout_route_count": 2,
                                                      "holdout_families": ["T7"]}),
                         source_prefix=SOURCE_PREFIX)  # fmt: skip
    with pytest.raises(QFError, match="unknown hard_case 'no_such_case'"):
        generate_records(facts, synthetic_config(hard=["no_such_case"]), source_prefix="x")
    with pytest.raises(QFError, match="uses prompt system_extract_v1"):
        generate_records(facts, synthetic_config(system_prompt_version="system_extract_v9"),
                         source_prefix="x")  # fmt: skip
    with pytest.raises(ValueError, match="must sum to 1"):
        synthetic_config(mix={"clean": 0.5, "hard": 0.4})
    with pytest.raises(ValueError, match="unknown LoadFacts fields"):
        synthetic_config(stratify_by=["revenue"])


def test_not_enough_loads_is_explicit_error() -> None:
    with pytest.raises(QFError, match="need 12 loads, only"):
        generate_records(facts_of(), small_config(pool={"train": 10, "val": 1, "test": 1,
                                                        "test_ood": 3, "smoke": 1}),
                         source_prefix="x")  # fmt: skip


def test_new_family_needs_only_registration() -> None:
    names = {name: TEMPLATE_FAMILIES.get(name)().language for name in TEMPLATE_FAMILIES.names()}
    assert names == {"T1": "ru", "T2": "ru", "T3": "ru", "T4": "en", "T5": "en", "T6": "en",
                     "T7": "ru", "T8": "en"}  # fmt: skip


# --- the stage and the CLI -----------------------------------------------------------------

RAW_TARGET = Path("data/raw/logistics-operations/54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07")


def install_project(project: Path, **config_changes: object) -> Path:
    shutil.copytree(FIXTURE, project / RAW_TARGET)
    (project / RAW_TARGET / "provenance.json").write_text(
        (REPO_ROOT / "tests/fixtures/provenance_mini.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_artifact(
        project / RAW_TARGET, "raw_dataset", "logistics_ops_csv_v1", "run-f", root=project
    )
    configs = project / "configs" / "data"
    configs.mkdir(parents=True)
    for name in ("source.yaml", "facts.yaml", "city_map_ru_v1.yaml"):
        shutil.copy(REPO_ROOT / "configs" / "data" / name, configs / name)
    config = small_config(**config_changes).model_dump(mode="json")
    (configs / "generate_v1.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    return configs / "generate_v1.yaml"


def test_cli_build_writes_splits_and_manifest(
    fake_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = install_project(fake_project)
    assert main(["data", "build"]) == 0
    out = capsys.readouterr().out
    assert "Built 10 load facts" in out and "train: 3 records" in out
    out_dir = Path("data/processed/generated_v1")
    refs = {name: read_artifact(out_dir / f"{name}.jsonl", "sft_dataset",
                                supported_versions("sft_dataset"))
            for name in ("train", "val", "test", "test_ood", "smoke")}  # fmt: skip
    facts_ref = load_artifact_manifest(Path("data/processed/load_facts.jsonl")).ref
    assert all(ref.parents == (facts_ref.sha256,) for ref in refs.values())
    records = {name: [SFTRecord.model_validate_json(line) for line in
                      (fake_project / ref.path).read_text(encoding="utf-8").splitlines()]
               for name, ref in refs.items()}  # fmt: skip
    assert {name: len(items) for name, items in records.items()} == {
        "train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2,
    }  # fmt: skip
    assert records["smoke"] == records["train"][:2]
    assert all(r.split == "train" for r in records["smoke"])
    assert all(r.source_id.startswith("yogape/logistics-operations@54e7d1d1a437:LOAD")
               for items in records.values() for r in items)  # fmt: skip
    extra = load_artifact_manifest(out_dir / "test_ood.jsonl").extra
    assert extra["holdout_families"] == ["T7", "T8"] and len(extra["holdout_routes"]) == 1
    manifest = read_manifest(fake_project / "runs" / refs["train"].producer_run_id)
    assert (manifest.kind, manifest.seed) == ("generate", 42)
    assert manifest.prompt_hashes == {"system_extract_v1": system_prompt_hash()}
    assert manifest.data_hashes["configs/data/generate_v1.yaml"]
    assert manifest.data_hashes["data/processed/load_facts.jsonl"] == facts_ref.sha256
    assert manifest.metrics["counts"]["test_ood"] == 3
    assert config_path.exists()


def test_cli_build_same_bytes_twice(fake_project: Path) -> None:
    install_project(fake_project)
    out = fake_project / "data/processed/generated_v1"
    assert main(["data", "build"]) == 0
    first = {p.name: p.read_bytes() for p in sorted(out.glob("*.jsonl"))}
    assert main(["data", "build"]) == 0
    assert first == {p.name: p.read_bytes() for p in sorted(out.glob("*.jsonl"))}


# --- real data (acceptance) ----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="raw dataset not fetched")
def test_invariants_on_real_data() -> None:
    facts = facts_of(REAL_RAW)
    cfg = repo_config()
    plan = sample_loads(facts, cfg)
    generated = [
        render_record(
            f,
            split,
            0,
            cfg,
            ood_reason=plan.ood_reasons.get(f.load_id),
            source_prefix=SOURCE_PREFIX,
        )  # fmt: skip
        for split, items in plan.splits.items()
        for f in items
    ]
    assert len(generated) == 2050
    by_id = {f.load_id: f for f in facts}
    money = money_of(REAL_RAW)
    for name in INVARIANTS:
        assert violations(name, generated, by_id, money) == [], name
    records, _ = generate_records(facts, cfg, source_prefix=SOURCE_PREFIX)
    assert sorted(g.record.id for g in generated) == sorted(
        r.id for items in records.values() for r in items
    )


def test_defensive_errors() -> None:
    facts = synthetic_facts()
    with pytest.raises(QFError, match="cannot hold out 20 of 12 routes"):
        sample_loads(facts, synthetic_config(ood={"holdout_route_count": 20,
                                                  "holdout_families": ["T7", "T8"]}))  # fmt: skip
    with pytest.raises(ValueError, match="non-negative"):
        synthetic_config(mix={"clean": 1.5, "hard": -0.5})
    with pytest.raises(QFError, match="no regular family for language en"):
        generate_records(facts, synthetic_config(families=["T1", "T2", "T3", "T7", "T8"]),
                         source_prefix="x")  # fmt: skip
    template = TEMPLATE_FAMILIES.get("T4")()
    russian = render_record(facts[0], "train", 0, synthetic_config(), ood_reason=None,
                            source_prefix="x", family="T1", kind="clean")  # fmt: skip
    with pytest.raises(QFError, match="family T4 renders en, not ru"):
        template.render(facts[0], russian.draft, Random(0))


class _NeverApplies:
    name = "never"

    def applicable(self, facts: LoadFacts) -> bool:
        return False

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: object) -> RequestDraft:
        raise AssertionError("not applicable")


class _AlwaysFails(_NeverApplies):
    name = "fails"

    def applicable(self, facts: LoadFacts) -> bool:
        return True

    def apply(self, draft: RequestDraft, facts: LoadFacts, rng: object) -> RequestDraft:
        raise HardCaseNotApplicable("no")


def test_hard_bucket_falls_back_to_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = HARD_CASES.get
    fakes = {"never": _NeverApplies, "fails": _AlwaysFails}
    monkeypatch.setattr(HARD_CASES, "get", lambda name: fakes.get(name) or real_get(name))
    cfg = synthetic_config(hard=["never", "fails"])
    g = render_record(synthetic_facts()[0], "train", 0, cfg, ood_reason=None, source_prefix="x",
                      kind="hard")  # fmt: skip
    assert g.draft.hard_cases == [] and g.record.variant.hard_cases == []


GENERATED = REPO_ROOT / "data" / "processed" / "generated_v1"


@pytest.mark.slow
@pytest.mark.skipif(not (GENERATED / "train.jsonl").exists(), reason="run `qf data build` first")
def test_e9_e10_on_generated_files() -> None:
    """The files `qf data build` wrote: held-out routes and families only in test_ood."""
    facts = {f.load_id: f for f in facts_of(REAL_RAW)}
    splits = {name: [SFTRecord.model_validate_json(line) for line in
                     (GENERATED / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in ("train", "val", "test", "test_ood")}  # fmt: skip
    extra = load_artifact_manifest(GENERATED / "train.jsonl", root=REPO_ROOT).extra
    held = set(extra["holdout_routes"])
    seen: set[str] = set()
    for name, records in splits.items():
        for record in records:
            load_id = record.group_id.removeprefix("load:")
            assert load_id not in seen, record.id
            seen.add(load_id)
            on_holdout = facts[load_id].route_id in held
            ood_family = record.template_family in extra["holdout_families"]
            reason = record.variant.ood_reason
            if name != "test_ood":
                assert not on_holdout and not ood_family and reason is None, record.id
            else:
                assert on_holdout == (reason in ("route", "both")), record.id
                assert ood_family == (reason in ("family", "both")), record.id
            assert check_record(record) == [], record.id
