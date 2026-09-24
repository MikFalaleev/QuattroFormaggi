"""The card_v2 generator (sub-step V3, D-094…D-096): config, sampling, hard-case minimums,
determinism, the build command and the review samples. Per-record invariants of every family
and hard case are in `tests/contracts/test_template_family_contract.py`."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from qf.cli.main import main
from qf.common import QFError, load_artifact_manifest
from qf.data import (
    REVIEW_SAMPLES_FILENAME,
    GenerateConfig,
    curate,
    generate_records,
    generate_rendered,
    render_review,
    sample_loads,
)
from qf.data.render import Fragment, agree
from qf.domain import check_record, load_system_prompt
from tests.conftest import REPO_ROOT
from tests.generation import (
    CONDITIONS_TABLE,
    INVARIANTS,
    REAL_RAW,
    SOURCE_PREFIX,
    facts_of,
    install_project,
    money_of,
    repo_config,
    repo_config_v2,
    synthetic_facts_v2,
    violations,
)

POOL = {"train": 150, "val": 20, "test": 60, "test_ood": 30, "smoke": 5}
OOD = {"holdout_route_count": 2, "holdout_families": ["T7", "T8"]}
SMALL_FLOOR = {"test": {"oversize_partial": 2, "conditions_scattered": 2}}


def small_v2(**changes: Any) -> GenerateConfig:
    fields: dict[str, Any] = {"pool": POOL, "ood": OOD, "hard_case_floor": SMALL_FLOOR,
                              "review_samples": 0}  # fmt: skip
    return repo_config_v2(**{**fields, **changes})


def digest(records: dict[str, list[Any]]) -> str:
    text = "".join(r.model_dump_json() + "\n" for items in records.values() for r in items)
    return hashlib.sha256(text.encode()).hexdigest()


# --- config --------------------------------------------------------------------------------


def test_committed_v2_config() -> None:
    cfg = repo_config_v2()
    assert (cfg.schema_version, cfg.system_prompt_version) == ("card_v2", "system_extract_v2")
    assert cfg.dataset_name == "generated_v2" and "conditions" in cfg.mix
    assert set(cfg.condition_hard) == {"condition_no_values", "required_condition_dropped",
                                       "oversize_partial", "conditions_scattered"}  # fmt: skip


@pytest.mark.parametrize(("changes", "message"), [
    ({"length_unit_share": None}, "needs length_unit_share"),
    ({"mix": {"clean": 0.6, "dropped_fields": 0.2, "hard": 0.2}}, "go together"),
    ({"condition_hard": {"condition_no_values": 0}}, "must be positive"),
    ({"stratify_by": ["color"]}, "unknown LoadFactsV2 fields"),
    ({"profile_floor": 1.5}, "less than 1"),
    ({"hard_case_floor": {"test": {"oversize_partial": 0}}}, "must be positive"),
    ({"length_unit_share": {"m": 0.5, "cm": 0.4}}, "must sum to 1"),
])  # fmt: skip
def test_v2_config_is_validated(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        repo_config_v2(**changes)


def test_v1_config_has_no_condition_settings() -> None:
    with pytest.raises(ValidationError, match="no special conditions to configure"):
        repo_config(length_unit_share={"m": 1.0})
    with pytest.raises(ValidationError, match="unknown LoadFacts fields"):
        repo_config(stratify_by=["profile"])


def test_unknown_names_fail_before_generation() -> None:
    with pytest.raises(QFError, match="unknown hard_case 'nope'"):
        generate_records(synthetic_facts_v2(), small_v2(hard_case_floor={"test": {"nope": 1}}),
                         source_prefix="x")  # fmt: skip


# --- sampling and hard-case minimums -------------------------------------------------------


def test_profile_floor_lifts_rare_profiles() -> None:
    facts = synthetic_facts_v2()
    natural = Counter(f.profile for f in facts)
    plan = sample_loads(facts, small_v2())
    sampled = [f for split in ("train", "val", "test") for f in plan.splits[split]]
    counts = Counter(f.profile for f in sampled)
    floor = small_v2().profile_floor or 0
    rare = [p for p, n in natural.items() if n / len(facts) < floor]
    assert rare  # the synthetic facts do have profiles below the floor
    unfloored = sample_loads(facts, small_v2(profile_floor=None))
    plain = Counter(f.profile for split in ("train", "val", "test")
                    for f in unfloored.splits[split])  # fmt: skip
    for profile in rare:  # every rare profile is present and not rarer than without the floor
        assert counts[profile] >= max(1, plain[profile]), profile
    assert sum(counts[p] for p in rare) > sum(plain[p] for p in rare)
    assert len(sampled) == sum(POOL[s] for s in ("train", "val", "test"))


def test_hard_case_floor_is_met_or_refused() -> None:
    records, _ = generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")
    test = Counter(case for r in records["test"] for case in r.variant.hard_cases)
    assert test["oversize_partial"] >= 2 and test["conditions_scattered"] >= 2
    impossible = small_v2(hard_case_floor={"test": {"oversize_partial": 10_000}})
    with pytest.raises(QFError, match="hard_case_floor asks for 10000"):
        generate_records(synthetic_facts_v2(), impossible, source_prefix="x")


def test_v2_records_are_valid_card_v2() -> None:
    records, _ = generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")
    prompt = load_system_prompt("system_extract_v2")
    all_records = [r for items in records.values() for r in items]
    assert all(r.schema_version == "card_v2" for r in all_records)
    assert all(r.messages[0].content == prompt for r in all_records)
    assert all(check_record(r) == [] for r in all_records)
    with_conditions = [r for r in all_records
                       if '"special_conditions":[]' not in r.messages[2].content]  # fmt: skip
    assert 0 < len(with_conditions) < len(all_records)


def test_equipment_sentences_agree_with_the_word() -> None:
    layout = (("Нужен {equipment}.", "ищу {equipment}", "{equipment} нужен"),
              ("We require a {equipment}.",))  # fmt: skip

    def word(text: str) -> dict[str, Fragment]:
        return {"equipment": Fragment(text)}

    assert agree(layout, word("фура"), "ru")[0] == (
        "Нужна {equipment}.", "нужна {equipment}", "{equipment} нужна")  # fmt: skip
    assert agree(layout, word("тент мега"), "ru") == layout  # «тент» is masculine
    assert agree(layout, word("insulated van"), "en")[1] == ("We require an {equipment}.",)
    assert agree(layout, word("dry van"), "en") == layout
    assert agree(layout, {}, "ru") == layout  # the equipment is not in the text


# --- determinism ---------------------------------------------------------------------------

_SCRIPT = """
import hashlib
from tests.generation import synthetic_facts_v2
from tests.test_generate_v2 import small_v2, digest
from qf.data import generate_records
records, _ = generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")
print(digest(records))
"""


def test_v2_generation_is_deterministic() -> None:
    reference = digest(generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")[0])
    facts = list(synthetic_facts_v2())
    random.Random(4).shuffle(facts)
    assert digest(generate_records(facts, small_v2(), source_prefix="x")[0]) == reference
    assert digest(generate_records(synthetic_facts_v2(), small_v2(seed=43),
                                   source_prefix="x")[0]) != reference  # fmt: skip
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO_ROOT, env=env,
                                capture_output=True, text=True, check=True)  # fmt: skip
        digests.add(result.stdout.strip())
    assert digests == {reference}


# --- review samples ------------------------------------------------------------------------


def test_curated_samples_cover_the_data() -> None:
    records, _ = generate_records(synthetic_facts_v2(), small_v2(), source_prefix="x")
    everything = [r for items in records.values() for r in items]
    chosen = curate(everything, seed=1, limit=30)
    assert len(chosen) == 30 and len({r.id for r in chosen}) == 30
    families = {r.template_family for r in everything}
    assert {r.template_family for r in chosen} == families
    hard = {",".join(r.variant.hard_cases) or "clean" for r in everything}
    assert {",".join(r.variant.hard_cases) or "clean" for r in chosen} == hard
    shuffled = everything[:]
    random.Random(9).shuffle(shuffled)
    assert [r.id for r in curate(shuffled, seed=1, limit=30)] == [r.id for r in chosen]
    text = render_review(records, chosen, "generated_v2")
    assert "## Трудные случаи по выборкам" in text and "**Особые условия:" in text
    assert text.count("```text") == 30


# --- the build command ---------------------------------------------------------------------


def test_build_v2_writes_generated_v2(
    fake_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_project(fake_project)
    configs = fake_project / "configs" / "data"
    shutil.copy(CONDITIONS_TABLE, configs / CONDITIONS_TABLE.name)
    pool = {"train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2}
    ood = {"holdout_route_count": 1, "holdout_families": ["T7", "T8"]}
    cfg = repo_config_v2(pool=pool, ood=ood, hard_case_floor={}, review_samples=5)
    path = configs / "generate_v2.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True))
    assert main(["data", "build", "--config", str(path)]) == 0
    out = fake_project / "data" / "processed" / "generated_v2"
    assert "load facts v2" in capsys.readouterr().out
    assert not (fake_project / "data" / "processed" / "generated_v1").exists()
    parents = load_artifact_manifest(out / "train.jsonl", root=fake_project).ref.parents
    facts_v2 = load_artifact_manifest(fake_project / "data/processed/load_facts_v2.jsonl",
                                      root=fake_project).ref  # fmt: skip
    assert parents == (facts_v2.sha256,)
    assert (out / REVIEW_SAMPLES_FILENAME).exists()
    first = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["schema_version"] == "card_v2"


# --- real data -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="raw dataset not fetched")
def test_invariants_on_real_v2_data() -> None:
    from qf.data import build_facts_v2, load_conditions_table

    facts = build_facts_v2(facts_of(REAL_RAW), load_conditions_table(CONDITIONS_TABLE))
    cfg = repo_config_v2()
    rendered, _ = generate_rendered(facts, cfg, source_prefix=SOURCE_PREFIX)
    generated = [g for items in rendered.values() for g in items]
    assert len(generated) == 2550
    by_id = {f.load_id: f for f in facts}
    money = money_of(REAL_RAW)
    for name in INVARIANTS:
        assert violations(name, generated, by_id, money) == [], name
    for split, minimums in cfg.hard_case_floor.items():
        counts = Counter(case for g in rendered[split] for case in g.record.variant.hard_cases)
        assert all(counts[case] >= n for case, n in minimums.items()), split
