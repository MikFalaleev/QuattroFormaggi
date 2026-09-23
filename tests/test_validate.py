"""Every check of `qf validate-data` on its own corrupted copy of a small generated dataset.

The corrupted files get a fresh artifact manifest (same parents and extra), so each case fails
for its own reason and not with a content-hash mismatch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qf.cli import main
from qf.common import QFError, load_artifact_manifest, read_artifact, write_artifact
from qf.contracts import SFTRecord, supported_versions
from qf.data import ValidationReport, run_validation, validate_dataset, write_sft_records
from qf.domain import load_system_prompt, parse_target, serialize_target
from tests.conftest import REPO_ROOT
from tests.factories import EXAMPLE_ANSWER_V2
from tests.generation import install_project

DATA = Path("data/processed/generated_v1")
Rows = list[dict[str, Any]]


@pytest.fixture
def project(fake_project: Path) -> Path:
    install_project(fake_project)
    assert main(["data", "build"]) == 0
    return fake_project


def rows(project: Path, name: str) -> Rows:
    path = project / DATA / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rewrite(project: Path, name: str, mutate: Callable[[Rows], Rows] | Rows) -> None:
    """Replace a split file and write its manifest again (same parents and extra)."""
    path = project / DATA / f"{name}.jsonl"
    manifest = load_artifact_manifest(path, root=project)
    new = mutate(rows(project, name)) if callable(mutate) else mutate
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in new)
    path.write_text(text, encoding="utf-8")
    write_artifact(path, "sft_dataset", manifest.ref.schema_version, manifest.ref.producer_run_id,
                   manifest.ref.parents, extra=manifest.extra, root=project)  # fmt: skip


def validate(project: Path, bench: Path | None = None) -> ValidationReport:
    paths = {name: project / DATA / f"{name}.jsonl"
             for name in ("train", "val", "test", "test_ood", "smoke")}  # fmt: skip
    return run_validation(paths, bench, root=project,
                          out_path=project / DATA / "validation_report.json").report  # fmt: skip


def found(report: ValidationReport) -> list[tuple[str, str]]:
    return sorted({(issue.record_id, issue.code) for issue in report.errors})


def edit_first(project: Path, name: str, change: Callable[[dict[str, Any]], None]) -> str:
    """Apply `change` to the first record of a split file; return its id (before the change)."""
    data = rows(project, name)
    record_id = str(data[0]["id"])
    change(data[0])
    rewrite(project, name, data)
    return record_id


# --- a clean dataset -----------------------------------------------------------------------


def test_clean_dataset_passes(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = validate(project)
    assert report.errors == [] and report.warnings == [] and report.counts == {}
    assert {name: info["records"] for name, info in report.datasets.items()} == {
        "train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2,
    }  # fmt: skip
    assert main(["validate-data"]) == 0
    assert "OK: 0 errors" in capsys.readouterr().out


# --- one check per corrupted dataset -------------------------------------------------------


def test_group_leak_detected(project: Path) -> None:
    train = rows(project, "train")
    copy = {**train[0], "id": "qf-val-copy-0", "split": "val"}
    user = {**copy["messages"][1], "content": copy["messages"][1]["content"] + " Срочно."}
    copy["messages"] = [copy["messages"][0], user, copy["messages"][2]]
    rewrite(project, "val", lambda data: [*data, copy])
    assert found(validate(project)) == [
        (train[0]["id"], "GROUP_LEAK"),
        ("qf-val-copy-0", "GROUP_LEAK"),
    ]


def test_exact_dup_across_splits_detected(project: Path) -> None:
    train = rows(project, "train")
    body = train[0]["messages"][1]["content"].split("\n\n", 1)[1]

    def same_text(record: dict[str, Any]) -> None:  # its own request date, the same text
        header = record["messages"][1]["content"].split("\n\n", 1)[0]
        record["messages"][1]["content"] = f"{header}\n\n{body.upper()}"

    val_id = edit_first(project, "val", same_text)
    assert found(validate(project)) == sorted(
        [(train[0]["id"], "DUP_EXACT"), (val_id, "DUP_EXACT")]
    )


def test_duplicate_inside_one_split_is_a_warning(project: Path) -> None:
    train = rows(project, "train")
    train[1]["messages"][1]["content"] = train[0]["messages"][1]["content"]
    rewrite(project, "train", train)
    smoke = rows(project, "smoke")
    rewrite(project, "smoke", [r for r in train if r["id"] in {s["id"] for s in smoke}])
    report = validate(project)
    assert report.errors == []
    assert {w.code for w in report.warnings} == {"DUP_EXACT_IN_SPLIT"}


def test_ood_route_leak_detected(project: Path) -> None:
    ood = rows(project, "test_ood")
    moved = next(r for r in ood if r["variant"]["ood_reason"] == "route")
    rewrite(project, "test_ood", [r for r in ood if r is not moved])
    moved = {**moved, "id": moved["id"].replace("test_ood", "train"), "split": "train",
             "variant": {**moved["variant"], "ood_reason": None}}  # fmt: skip
    rewrite(project, "train", lambda data: [*data, moved])
    assert found(validate(project)) == [(moved["id"], "OOD_ROUTE_LEAK")]


def test_ood_family_leak_detected(project: Path) -> None:
    record_id = edit_first(project, "val", lambda r: r.update(template_family="T7"))
    assert found(validate(project)) == [(record_id, "OOD_FAMILY_LEAK")]


def test_bench_leak_detected(project: Path) -> None:
    train_record = SFTRecord.model_validate_json(
        (project / DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    bench = project / "data" / "splits" / "bench_v1.jsonl"
    bench.parent.mkdir(parents=True)
    write_sft_records([train_record], bench, run_id="run-b", parents=[], root=project,
                      kind="benchmark")  # fmt: skip
    report = validate(project, bench)
    assert found(report) == [(train_record.id, "BENCH_LEAK")]
    assert report.bench and report.bench["records"] == 1


def test_inconsistent_missing_detected(project: Path) -> None:
    def wrong_missing(record: dict[str, Any]) -> None:
        target = parse_target(record["messages"][2]["content"])
        wrong = target.model_copy(update={"missing_fields": ["origin", *target.missing_fields]})
        record["messages"][2]["content"] = serialize_target(wrong)

    record_id = edit_first(project, "val", wrong_missing)
    assert found(validate(project)) == [(record_id, "TARGET_INCONSISTENT")]


def test_non_canonical_detected(project: Path) -> None:
    def pretty(record: dict[str, Any]) -> None:
        answer = json.loads(record["messages"][2]["content"])
        record["messages"][2]["content"] = json.dumps(answer, ensure_ascii=False, indent=2)

    record_id = edit_first(project, "val", pretty)
    assert found(validate(project)) == [(record_id, "NOT_CANONICAL")]


def test_prompt_mismatch_detected(project: Path) -> None:
    def edit(record: dict[str, Any]) -> None:
        record["messages"][0]["content"] += "\nОтвечай кратко."

    record_id = edit_first(project, "test", edit)
    assert found(validate(project)) == [(record_id, "PROMPT_MISMATCH")]


def test_card_v2_record_is_checked_with_its_prompt(project: Path) -> None:
    """A card_v2 record is checked by the rules of card_v2 and against prompt v2 (D-086)."""

    def to_v2(prompt: str) -> Callable[[dict[str, Any]], None]:
        def edit(record: dict[str, Any]) -> None:
            record["schema_version"] = "card_v2"
            record["messages"][0]["content"] = load_system_prompt(prompt)
            record["messages"][2]["content"] = EXAMPLE_ANSWER_V2

        return edit

    edit_first(project, "test", to_v2("system_extract_v2"))
    assert found(validate(project)) == []
    record_id = edit_first(project, "test", to_v2("system_extract_v1"))
    assert found(validate(project)) == [(record_id, "PROMPT_MISMATCH")]


def test_role_order_and_schema_detected(project: Path) -> None:
    data = rows(project, "val")
    data[0]["messages"] = [data[0]["messages"][1], data[0]["messages"][0], data[0]["messages"][2]]
    rewrite(project, "val", data)
    test = rows(project, "test")
    test[0]["comment"] = "extra key"
    rewrite(project, "test", test)
    assert found(validate(project)) == sorted([(data[0]["id"], "ROLE_ORDER"),
                                               (test[0]["id"], "SCHEMA")])  # fmt: skip


def test_unknown_task_detected(project: Path) -> None:
    record_id = edit_first(project, "val", lambda r: r.update(task="no_such_task"))
    assert found(validate(project)) == [(record_id, "UNKNOWN_TASK")]


def test_duplicate_id_and_split_mismatch_detected(project: Path) -> None:
    train_id = rows(project, "train")[0]["id"]
    test_id = edit_first(project, "test", lambda r: r.update(split="val"))
    edit_first(project, "val", lambda r: r.update(id=train_id))
    assert found(validate(project)) == sorted([(train_id, "DUP_ID"), (test_id, "SPLIT_MISMATCH")])


def test_smoke_must_repeat_train(project: Path) -> None:
    def edit(record: dict[str, Any]) -> None:
        record["messages"][1]["content"] += " Срочно."

    record_id = edit_first(project, "smoke", edit)
    assert found(validate(project)) == [(record_id, "SMOKE_NOT_IN_TRAIN")]


def test_holdout_mismatch_and_unknown_load_detected(project: Path) -> None:
    path = project / DATA / "test.jsonl"
    manifest = load_artifact_manifest(path, root=project)
    extra = {**manifest.extra, "holdout_routes": ["RTE09999"]}
    write_artifact(path, "sft_dataset", "sft_record_v1", "run-x", manifest.ref.parents,
                   extra=extra, root=project)  # fmt: skip
    record_id = edit_first(project, "val", lambda r: r.update(group_id="load:NOPE"))
    codes = found(validate(project))
    assert ("<manifests>", "HOLDOUT_MISMATCH") in codes
    assert (record_id, "UNKNOWN_LOAD") in codes


# --- failures that stop the validation -----------------------------------------------------


def test_tampered_file_is_refused(project: Path) -> None:
    path = project / DATA / "train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(QFError, match="content hash mismatch"):
        validate(project)


def test_missing_split_or_lineage_is_refused(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "data/processed/load_facts.jsonl.manifest.json").unlink()
    with pytest.raises(QFError, match="parent artifact .* not found"):
        validate(project)
    (project / DATA / "val.jsonl").unlink()
    assert main(["validate-data"]) == 1
    assert "missing split files val" in capsys.readouterr().err


def test_report_contains_record_id_and_reason(project: Path) -> None:
    record_id = edit_first(project, "val", lambda r: r.update(task="no_such_task"))
    assert main(["validate-data"]) == 1
    stored = json.loads((project / DATA / "validation_report.json").read_text(encoding="utf-8"))
    message = "unknown task 'no_such_task'; known: shipment_extraction"
    assert stored["errors"] == [
        {"record_id": record_id, "split": "val", "code": "UNKNOWN_TASK", "message": message}
    ]
    manifest = load_artifact_manifest(DATA / "validation_report.json")
    assert (manifest.ref.kind, manifest.ref.schema_version) == ("metrics", "validation_report_v1")
    assert len(manifest.ref.parents) == 5


# --- real data (acceptance) ----------------------------------------------------------------

REAL = REPO_ROOT / DATA


@pytest.mark.slow
@pytest.mark.skipif(not (REAL / "train.jsonl").exists(), reason="run `qf data build` first")
def test_validate_real_generated() -> None:
    """Read-only: verifies the artifacts and validates, writes nothing into the repository."""
    refs = {name: read_artifact(REAL / f"{name}.jsonl", "sft_dataset",
                                supported_versions("sft_dataset"), root=REPO_ROOT)
            for name in ("train", "val", "test", "test_ood", "smoke")}  # fmt: skip
    report = validate_dataset(refs, None, root=REPO_ROOT)
    assert report.errors == [] and report.warnings == []
    assert report.datasets["train"]["records"] == 1500
