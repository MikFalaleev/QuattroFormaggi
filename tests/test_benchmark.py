from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from qf.cli import main
from qf.common import (
    DataValidationError,
    QFError,
    load_artifact_manifest,
    load_yaml_config,
    read_artifact,
    sha256_file,
    write_artifact,
)
from qf.contracts import SFTRecord, supported_versions
from qf.data import (
    BenchmarkConfig,
    generate_dataset,
    load_manual_cases,
    read_sft_records,
    save_facts,
    select_candidates,
    verify_benchmark,
)
from tests.conftest import REPO_ROOT
from tests.generation import repo_config, synthetic_facts

GENERATED = Path("data/processed/generated_v1")
SPLITS = Path("data/splits")
CSV_PATH = Path("configs/eval/bench_v1_review.csv")
HARD = ("per_piece_weight", "conflict_weight", "conflict_pieces", "relative_date",
        "distractor_numbers", "city_lang_switch")  # fmt: skip
SMALL_SLICES = [
    {"name": "clean_id", "count": 6, "splits": ["test"], "kind": "clean",
     "stratify_by": ["template_family", "language"]},
    {"name": "clean_ood", "count": 4, "splits": ["test_ood"], "kind": "clean",
     "stratify_by": ["ood_reason"]},
    {"name": "missing_fields", "count": 6, "splits": ["test", "test_ood"],
     "kind": "dropped_fields"},
    *({"name": kind, "count": 1, "reserve": 1, "splits": ["test", "test_ood"], "kind": kind}
      for kind in HARD),
]  # fmt: skip


@pytest.fixture
def project(fake_project: Path) -> Path:
    """A generated dataset big enough for small slices, and a benchmark config."""
    root = fake_project
    facts_path = root / "data" / "processed" / "load_facts.jsonl"
    facts_path.parent.mkdir(parents=True)
    save_facts(synthetic_facts(), facts_path, run_id="run-f", parents=[], root=root)
    cfg = repo_config(pool={"train": 100, "val": 20, "test": 150, "test_ood": 90, "smoke": 10},
                      ood={"holdout_route_count": 2, "holdout_families": ["T7", "T8"]})  # fmt: skip
    config_path = root / "configs" / "data" / "generate_v1.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True))
    generate_dataset(facts_path, cfg, root=root, out_dir=root / GENERATED,
                     source_prefix="test@000000000000", config_path=config_path)  # fmt: skip
    bench = yaml.safe_load((REPO_ROOT / "configs/eval/benchmark_v1.yaml").read_text())
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "configs/eval/benchmark_v1.yaml").write_text(
        yaml.safe_dump({**bench, "slices": SMALL_SLICES}, allow_unicode=True)
    )
    return root


def config(root: Path) -> BenchmarkConfig:
    return load_yaml_config(root / "configs/eval/benchmark_v1.yaml", BenchmarkConfig)


def csv_rows(root: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((root / CSV_PATH).read_text(encoding="utf-8-sig"))))


def fill(root: Path, verdicts: dict[str, str] | None = None, default: str = "ok",
         delimiter: str = ",", comments: dict[str, str] | None = None) -> None:  # fmt: skip
    """The CSV as a person would save it: verdicts (any case), comments, chosen delimiter."""
    rows = csv_rows(root)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter=delimiter)
    writer.writeheader()
    for row in rows:
        row["verdict"] = (verdicts or {}).get(row["id"], default)
        row["comment"] = (comments or {}).get(row["id"], "")
        writer.writerow(row)
    (root / CSV_PATH).write_text("﻿" + buffer.getvalue(), encoding="utf-8")


def bench_records(root: Path) -> list[SFTRecord]:
    return read_sft_records(root / SPLITS / "bench_v1.jsonl")[0]


# --- selection and export ------------------------------------------------------------------


def test_benchmark_composition_matches_config(project: Path) -> None:
    assert main(["bench", "export-review"]) == 0
    rows = csv_rows(project)
    counts = Counter((row["slice"], row["role"]) for row in rows)
    for spec in SMALL_SLICES:
        assert counts[(spec["name"], "main")] == spec["count"]
        assert counts[(spec["name"], "reserve")] == spec.get("reserve", 0)
    assert all(row["verdict"] == "" for row in rows)
    assert (project / CSV_PATH).read_bytes().startswith(b"\xef\xbb\xbf")
    candidates = read_sft_records(project / SPLITS / "bench_v1" / "candidates.jsonl")[0]
    assert len({r.id for r in candidates}) == len(rows) == len(candidates)
    assert {r.split for r in candidates} <= {"test", "test_ood"}
    review = (project / SPLITS / "bench_v1" / "review_bench_v1.md").read_text(encoding="utf-8")
    assert all(row["id"] in review for row in rows) and "Недостающие" in review


def test_selection_is_deterministic_and_stratified(project: Path) -> None:
    datasets = {
        s: read_sft_records(project / GENERATED / f"{s}.jsonl")[0] for s in ("test", "test_ood")
    }
    first = select_candidates(datasets, config(project))
    assert [c.record.id for c in first] == [
        c.record.id for c in select_candidates(datasets, config(project))
    ]
    reasons = {c.record.variant.ood_reason for c in first if c.slice == "clean_ood"}
    assert reasons == {"route", "family", "both"}


def test_export_refuses_to_overwrite_verdicts(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bench", "export-review"]) == 0
    fill(project, default="")
    assert main(["bench", "export-review"]) == 0  # no verdict yet: a new export is fine
    first = csv_rows(project)[0]["id"]
    fill(project, {first: "ok"}, default="")
    assert main(["bench", "export-review"]) == 1
    assert "already has verdicts" in capsys.readouterr().err


# --- review --------------------------------------------------------------------------------


def test_import_review_ok_drop_fix(project: Path) -> None:
    assert main(["bench", "export-review"]) == 0
    ids = [row["id"] for row in csv_rows(project)]
    fill(project, {ids[0]: " FIX ", ids[1]: "Drop"}, default="OK", delimiter=";",
         comments={ids[0]: "вес в тексте не совпадает"})  # fmt: skip
    assert main(["bench", "import-review"]) == 0
    reviewed = read_sft_records(project / SPLITS / "bench_v1" / "reviewed.jsonl")[0]
    assert {r.id for r in reviewed} == set(ids[2:])
    assert all(r.reviewed for r in reviewed)
    fixes = (project / SPLITS / "bench_v1" / "fix_list.md").read_text(encoding="utf-8")
    assert ids[0] in fixes and "вес в тексте не совпадает" in fixes and ids[1] not in fixes


def test_import_rejects_unknown_verdict_and_id(project: Path) -> None:
    assert main(["bench", "export-review"]) == 0
    fill(project, default="maybe")
    with pytest.raises(DataValidationError, match="BAD_VERDICT"):
        from qf.data import read_verdicts

        read_verdicts(project / CSV_PATH)
    assert main(["bench", "import-review"]) == 1
    fill(project)
    text = (project / CSV_PATH).read_text(encoding="utf-8-sig")
    (project / CSV_PATH).write_text(text + "999,qf-test-NOPE-0,clean_id,main,ok,\n")
    assert main(["bench", "import-review"]) == 1


def test_agent_cannot_set_reviewed_without_csv(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bench", "export-review"]) == 0
    assert main(["bench", "import-review"]) == 0  # no verdicts at all
    assert main(["bench", "freeze"]) == 1
    assert "have no verdict yet" in capsys.readouterr().err
    assert not (project / SPLITS / "bench_v1.jsonl").exists()
    assert main(["bench", "freeze", "--allow-unreviewed"]) == 0
    assert "unreviewed record(s) frozen" in capsys.readouterr().err
    extra = load_artifact_manifest(SPLITS / "bench_v1.jsonl").extra
    assert extra["warnings"] and not any(r.reviewed for r in bench_records(project))


# --- freeze --------------------------------------------------------------------------------


def _review_all(project: Path, verdicts: dict[str, str] | None = None) -> list[dict[str, str]]:
    assert main(["bench", "export-review"]) == 0
    rows = csv_rows(project)
    fill(project, verdicts)
    assert main(["bench", "import-review"]) == 0
    return rows


def test_freeze_uses_reserve_and_reports_shortfall(project: Path) -> None:
    rows = _review_all(project)
    piece = [r["id"] for r in rows if r["slice"] == "per_piece_weight"]
    conflict = [r["id"] for r in rows if r["slice"] == "conflict_weight"]
    (project / CSV_PATH).unlink()
    (project / SPLITS / "bench_v1").rename(project / "old_work")
    rows = _review_all(project, {piece[0]: "drop", conflict[0]: "fix", conflict[1]: "drop"})
    cfg = config(project)
    assert main(["bench", "freeze"]) == 0
    frozen = {r.id for r in bench_records(project)}
    assert piece[1] in frozen and piece[0] not in frozen  # the reserve replaced the dropped one
    assert not frozen & set(conflict)
    extra = load_artifact_manifest(SPLITS / "bench_v1.jsonl").extra
    assert extra["shortfalls"] == {"conflict_weight": 1}
    assert sum(extra["slices"].values()) == sum(s.count for s in cfg.slices) - 1
    assert all(r.split == "bench" and r.reviewed for r in bench_records(project))


def test_bench_records_removed_from_test_pools(project: Path) -> None:
    before = {p.name: sha256_file(p) for p in (project / GENERATED).glob("*.jsonl")}
    rows = _review_all(project)
    assert main(["bench", "freeze"]) == 0
    frozen = {r.id for r in bench_records(project)}
    reserve = {r["id"] for r in rows if r["role"] == "reserve"}
    for name in ("test", "test_ood"):
        left = {r.id for r in read_sft_records(project / SPLITS / f"{name}.jsonl")[0]}
        original = {r.id for r in read_sft_records(project / GENERATED / f"{name}.jsonl")[0]}
        assert left == original - frozen
    left_all = {r.id for n in ("test", "test_ood")
                for r in read_sft_records(project / SPLITS / f"{n}.jsonl")[0]}  # fmt: skip
    assert reserve <= left_all  # unused reserve stays in the test pools
    assert before == {p.name: sha256_file(p) for p in (project / GENERATED).glob("*.jsonl")}
    train = read_artifact(SPLITS / "train.jsonl", "sft_dataset", supported_versions("sft_dataset"))
    source = load_artifact_manifest(GENERATED / "train.jsonl").ref
    assert train.sha256 == source.sha256 and train.parents == source.parents  # an exact copy
    test = read_artifact(SPLITS / "test.jsonl", "sft_dataset", supported_versions("sft_dataset"))
    assert test.parents == (sha256_file(project / GENERATED / "test.jsonl"),)
    extra = load_artifact_manifest(SPLITS / "test_ood.jsonl").extra
    assert extra["holdout_families"] == ["T7", "T8"] and extra["without"] == "bench_v1"


def test_freeze_is_once_and_verify_detects_changes(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _review_all(project)
    assert main(["bench", "freeze"]) == 0
    assert main(["bench", "verify"]) == 0
    sha_text = (project / SPLITS / "bench_v1.sha256").read_text()
    assert sha_text == f"{sha256_file(project / SPLITS / 'bench_v1.jsonl')}  bench_v1.jsonl\n"
    assert main(["bench", "freeze"]) == 1
    assert main(["bench", "export-review"]) == 1
    assert "bench_v2" in capsys.readouterr().err
    path = project / SPLITS / "bench_v1.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        verify_benchmark(path, root=project)
    assert main(["bench", "verify"]) == 1


def test_validate_after_freeze_and_bench_leak_into_train(project: Path) -> None:
    _review_all(project)
    assert main(["bench", "freeze"]) == 0
    args = ["validate-data", "--data-dir", "data/splits", "--bench", "data/splits/bench_v1.jsonl"]
    assert main(args) == 0
    leaked = bench_records(project)[0]
    train_path = project / SPLITS / "train.jsonl"
    manifest = load_artifact_manifest(train_path, root=project)
    copy = SFTRecord.model_validate({**leaked.model_dump(), "split": "train",
                                     "id": leaked.id + "-copy"})  # fmt: skip
    train_path.write_text(train_path.read_text(encoding="utf-8") + copy.model_dump_json() + "\n",
                          encoding="utf-8")  # fmt: skip
    write_artifact(train_path, "sft_dataset", "sft_record_v1", "run-x", manifest.ref.parents,
                   extra=manifest.extra, root=project)  # fmt: skip
    assert main(args) == 1
    report = json.loads((project / SPLITS / "validation_report.json").read_text(encoding="utf-8"))
    assert {"record_id": leaked.id, "split": "bench", "code": "BENCH_LEAK",
            "message": "group also in train"} in report["errors"]  # fmt: skip


# --- hand-written cases --------------------------------------------------------------------

CARD = {
    "shipper_name": "ООО «Ромашка»", "cargo_category": "food_beverage",
    "equipment_type": "reefer", "pieces": 18, "weight_total": {"value": 12.5, "unit": "t"},
    "weight_per_piece": None, "origin": {"city": "Казань", "region": "Республика Татарстан"},
    "destination": {"city": "Москва", "region": "Москва"}, "pickup_date": "2026-09-22",
    "delivery_date": None, "temperature_c": None,
}  # fmt: skip


def write_manual(path: Path, cases: list[dict[str, object]]) -> Path:
    path.write_text(yaml.safe_dump(cases, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_manual_cases_become_checked_records(tmp_path: Path) -> None:
    outside = {**CARD, "destination": {"city": "Тверь", "region": "Тверская область"}}
    path = write_manual(tmp_path / "manual.yaml", [
        {"author": "ivanov", "language": "ru", "request_date": "2026-09-20",
         "text": "Нужен реф из Казани в Москву, 18 паллет, 12,5 т, загрузка 22.09.2026."},
        {"author": "ivanov", "language": "ru", "request_date": "2026-09-20",
         "text": "Из Казани в Тверь, 12,5 т.", "card": outside},
    ])  # fmt: skip
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases[0]["card"] = CARD
    write_manual(path, cases)
    records, warnings = load_manual_cases(path)
    assert [r.id for r in records] == ["manual-ivanov-1", "manual-ivanov-2"]
    first = records[0]
    assert (first.synthetic, first.reviewed, first.split, first.source_id) == (
        False, True, "bench", "manual:ivanov:1",
    )  # fmt: skip
    assert json.loads(first.messages[2].content)["missing_fields"] == ["temperature_c"]
    assert first.messages[1].content.startswith("Дата запроса: 2026-09-20\n\n")
    assert warnings == ["manual-ivanov-2: destination 'Тверь' is not one of the 20 catalog cities"]


def test_manual_cases_are_validated(tmp_path: Path) -> None:
    wrong_region = {**CARD, "origin": {"city": "Казань", "region": "Татарстан"}}
    path = write_manual(tmp_path / "bad.yaml", [
        {"author": "ivanov", "language": "ru", "request_date": "2026-09-20", "text": "x",
         "card": wrong_region},
        {"author": "Иван", "language": "ru", "request_date": "2026-09-20", "text": "y",
         "card": CARD},
        {"author": "ivanov", "language": "ru", "request_date": "2026-09-20", "text": "z",
         "card": {**CARD, "pieces": "18"}},
    ])  # fmt: skip
    with pytest.raises(DataValidationError) as caught:
        load_manual_cases(path)
    assert [(i.record_id, i.code) for i in caught.value.issues] == [
        ("manual-ivanov-1", "REGION"), ("bad.yaml#2", "SCHEMA"), ("bad.yaml#3", "SCHEMA"),
    ]  # fmt: skip


def test_freeze_with_manual_cases(project: Path) -> None:
    _review_all(project)
    path = write_manual(project / "configs" / "eval" / "bench_v1_manual.yaml", [
        {"author": "ivanov", "language": "ru", "request_date": "2026-09-20",
         "text": "Нужен реф из Казани в Москву, 18 паллет, 12,5 т, загрузка 22.09.2026.",
         "card": CARD},
    ])  # fmt: skip
    assert main(["bench", "freeze", "--manual", str(path)]) == 0
    manual = [r for r in bench_records(project) if not r.synthetic]
    assert [r.id for r in manual] == ["manual-ivanov-1"]
    extra = load_artifact_manifest(SPLITS / "bench_v1.jsonl").extra
    assert extra["manual"] == 1
    ref = load_artifact_manifest(SPLITS / "bench_v1.jsonl").ref
    assert sha256_file(path) in ref.parents


# --- real data -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not (REPO_ROOT / GENERATED / "test.jsonl").exists(), reason="no dataset")
def test_real_slices_can_be_filled() -> None:
    cfg = load_yaml_config(REPO_ROOT / "configs/eval/benchmark_v1.yaml", BenchmarkConfig)
    datasets = {}
    for name in ("test", "test_ood"):
        ref = read_artifact(REPO_ROOT / GENERATED / f"{name}.jsonl", "sft_dataset",
                            supported_versions("sft_dataset"), root=REPO_ROOT)  # fmt: skip
        datasets[name] = read_sft_records(REPO_ROOT / ref.path)[0]
    candidates = select_candidates(datasets, cfg)
    for spec in cfg.slices:
        mains = sum(c.slice == spec.name and c.role == "main" for c in candidates)
        assert mains == spec.count, spec.name
    assert 188 <= len(candidates) <= 200


def test_manual_case_variants_and_bad_files(tmp_path: Path) -> None:
    per_piece = {**CARD, "weight_total": None, "weight_per_piece": {"value": 700, "unit": "kg"},
                 "origin": None}  # fmt: skip
    path = write_manual(tmp_path / "ok.yaml", [
        {"author": "ivanov", "language": "en", "request_date": "2026-09-20",
         "text": "Reefer to Moscow, 18 pallets, 700 kg each.", "card": per_piece,
         "conflicts": [{"field": "origin", "values": ["Казань", "Самара"]}]},
    ])  # fmt: skip
    (record,), _ = load_manual_cases(path)
    assert (record.variant.weight_mode, record.variant.weight_unit) == ("per_piece", "kg")
    assert json.loads(record.messages[2].content)["missing_fields"] == ["origin", "temperature_c"]
    (tmp_path / "list.yaml").write_text("author: ivanov\n", encoding="utf-8")
    with pytest.raises(QFError, match="must be a list"):
        load_manual_cases(tmp_path / "list.yaml")
    (tmp_path / "broken.yaml").write_text("- author: [\n", encoding="utf-8")
    with pytest.raises(QFError, match="cannot read manual cases"):
        load_manual_cases(tmp_path / "broken.yaml")


def test_manual_example_file_is_valid() -> None:
    records, warnings = load_manual_cases(REPO_ROOT / "configs/eval/bench_v1_manual.example.yaml")
    assert [r.id for r in records] == ["manual-example-1"] and warnings == []
    assert json.loads(records[0].messages[2].content)["missing_fields"] == ["temperature_c"]
