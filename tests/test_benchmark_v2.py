"""bench_v2: the card_v2 benchmark with condition and equipment slices (sub-step V5)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from qf.cli import main
from qf.common import DataValidationError, QFError, load_yaml_config, write_artifact
from qf.data import (
    BenchmarkConfig,
    export_review,
    freeze,
    generate_dataset,
    import_review_stage,
    load_manual_cases,
    read_sft_records,
    select_candidates,
)
from qf.domain import check_record, get_target_schema
from tests.conftest import REPO_ROOT
from tests.generation import REAL_RAW, repo_config_v2, synthetic_facts_v2

GENERATED = Path("data/processed/generated_v2")
OUT = Path("data/splits_v2")
CSV_PATH = Path("configs/eval/bench_v2_review.csv")
EXAMPLE = REPO_ROOT / "configs" / "eval" / "bench_v2_manual.example.yaml"
SLICES: list[dict[str, Any]] = [
    {"name": "conditions_scattered", "count": 2, "reserve": 1, "splits": ["test", "test_ood"],
     "kind": "conditions_scattered"},
    {"name": "cond_oversize", "count": 2, "splits": ["test", "test_ood"], "kind": "clean",
     "condition": "oversize"},
    {"name": "equip_reefer", "count": 3, "splits": ["test", "test_ood"], "kind": "clean",
     "equipment": "reefer"},
    {"name": "clean_id", "count": 6, "splits": ["test"], "kind": "clean",
     "stratify_by": ["language"]},
]  # fmt: skip


@pytest.fixture
def project(fake_project: Path) -> Path:
    """generated_v2 of the synthetic facts and a small bench_v2 config."""
    root = fake_project
    facts_path = root / "data" / "processed" / "load_facts_v2.jsonl"
    facts_path.parent.mkdir(parents=True)
    facts_path.write_text("".join(f.model_dump_json() + "\n" for f in synthetic_facts_v2()),
                          encoding="utf-8")  # fmt: skip
    write_artifact(facts_path, "load_facts", "load_facts_v2", "run-f", [], root=root)
    pool = {"train": 100, "val": 20, "test": 150, "test_ood": 90, "smoke": 10}
    ood = {"holdout_route_count": 2, "holdout_families": ["T7", "T8"]}
    cfg = repo_config_v2(pool=pool, ood=ood, hard_case_floor={"test": {"conditions_scattered": 3}},
                         review_samples=0)  # fmt: skip
    config_path = root / "configs" / "data" / "generate_v2.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True))
    generate_dataset(facts_path, cfg, root=root, out_dir=root / GENERATED,
                     source_prefix="test@000000000000", config_path=config_path)  # fmt: skip
    bench = yaml.safe_load((REPO_ROOT / "configs/eval/benchmark_v2.yaml").read_text())
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs/eval/benchmark_v2.yaml").write_text(
        yaml.safe_dump({**bench, "slices": SLICES}, allow_unicode=True)
    )
    return root


def config(root: Path) -> BenchmarkConfig:
    return load_yaml_config(root / "configs/eval/benchmark_v2.yaml", BenchmarkConfig)


def fill_ok(root: Path) -> None:
    text = (root / CSV_PATH).read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "verdict": "ok"})
    (root / CSV_PATH).write_text(buffer.getvalue(), encoding="utf-8")


def _card(record: Any) -> Any:
    return get_target_schema(record.schema_version).parse(record.messages[2].content).card


def test_slices_filter_by_condition_and_equipment(project: Path) -> None:
    outcome = export_review(config(project), root=project)
    by_slice: dict[str, list[Any]] = {}
    for candidate in outcome.candidates:
        by_slice.setdefault(candidate.slice, []).append(candidate.record)
    assert all("oversize" in {c.kind for c in _card(r).special_conditions}
               for r in by_slice["cond_oversize"])  # fmt: skip
    assert all(_card(r).equipment_type == "reefer" for r in by_slice["equip_reefer"])
    assert all(r.variant.hard_cases == [] for name in ("cond_oversize", "equip_reefer", "clean_id")
               for r in by_slice[name])  # fmt: skip
    assert all(r.schema_version == "card_v2" for r in by_slice["clean_id"])
    markdown = outcome.markdown_path.read_text(encoding="utf-8")
    assert "**Особые условия:" in markdown and "Температура:" not in markdown


def test_review_and_freeze_bench_v2_with_manual_cases(project: Path) -> None:
    cfg = config(project)
    export_review(cfg, root=project)
    fill_ok(project)
    import_review_stage(cfg, project / CSV_PATH, root=project)
    manual, _ = load_manual_cases(EXAMPLE, "card_v2")
    outcome = freeze(cfg, root=project, manual=manual, manual_source=EXAMPLE)
    assert outcome.bench.path.as_posix() == "data/splits_v2/bench_v2.jsonl"
    assert outcome.manual == 4 and outcome.shortfalls == {}
    frozen, issues = read_sft_records(project / OUT / "bench_v2.jsonl")
    assert issues == [] and all(r.schema_version == "card_v2" for r in frozen)
    assert all(check_record(r) == [] for r in frozen)
    assert (project / OUT / "train.jsonl").exists() and not (project / "data/splits").exists()
    assert main(["validate-data", "--data-dir", str(project / OUT),
                 "--bench", str(project / OUT / "bench_v2.jsonl")]) == 0  # fmt: skip


def test_schemas_do_not_mix(project: Path) -> None:
    cfg = config(project)
    manual_v1, _ = load_manual_cases(REPO_ROOT / "configs/eval/bench_v1_manual.example.yaml")
    export_review(cfg, root=project)
    fill_ok(project)
    import_review_stage(cfg, project / CSV_PATH, root=project)
    with pytest.raises(QFError, match="are not card_v2"):
        freeze(cfg, root=project, manual=manual_v1)
    with pytest.raises(DataValidationError):  # a card_v1 card is not a card_v2 card
        load_manual_cases(REPO_ROOT / "configs/eval/bench_v1_manual.example.yaml", "card_v2")
    v1_slices = {**cfg.model_dump(mode="json"), "schema_version": "card_v1"}
    with pytest.raises(ValidationError, match="need card_v2 slices"):
        BenchmarkConfig.model_validate(v1_slices)


def test_slice_keys_must_differ() -> None:
    base = yaml.safe_load((REPO_ROOT / "configs/eval/benchmark_v2.yaml").read_text())
    twice = {**base, "slices": [SLICES[1], {**SLICES[1], "name": "cond_oversize_2"}]}
    with pytest.raises(ValidationError, match="must be unique"):
        BenchmarkConfig.model_validate(twice)


def test_manual_v2_cases_are_canonical(tmp_path: Path) -> None:
    records, warnings = load_manual_cases(EXAMPLE, "card_v2")
    assert warnings == [] and len(records) == 4
    assert all(check_record(r) == [] for r in records)
    first = _card(records[0])
    assert [c.kind for c in first.special_conditions] == ["temperature", "sensors"]
    second = _card(records[1])
    assert second.special_conditions[0].methods == ["chains", "wheel_chocks"]
    third = get_target_schema("card_v2").parse(records[2].messages[2].content)
    assert third.missing_fields == ["special_conditions.packaging"]
    bad = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))[:1]
    bad[0]["card"]["special_conditions"] = [{"kind": "adr", "class": 3}]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_manual_cases(path, "card_v2")


@pytest.mark.slow
@pytest.mark.skipif(
    not (REPO_ROOT / GENERATED / "test.jsonl").exists() or not REAL_RAW.exists(),
    reason="run `qf data build --config configs/data/generate_v2.yaml` first",
)
def test_committed_bench_v2_config_fills_every_slice() -> None:
    cfg = load_yaml_config(REPO_ROOT / "configs/eval/benchmark_v2.yaml", BenchmarkConfig)
    data = {split: read_sft_records(REPO_ROOT / GENERATED / f"{split}.jsonl")[0]
            for split in ("test", "test_ood")}  # fmt: skip
    candidates = select_candidates(data, cfg)
    for spec in cfg.slices:
        mains = sum(c.slice == spec.name and c.role == "main" for c in candidates)
        spares = sum(c.slice == spec.name and c.role == "reserve" for c in candidates)
        assert (mains, spares) == (spec.count, spec.reserve), spec.name
