from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from qf.cli import main
from qf.common import QFError, load_artifact_manifest, read_manifest
from qf.contracts import SFTRecord, Splitter
from qf.data import (
    SPLITTERS,
    GroupHashSplitter,
    assign_splits,
    read_sft_records,
    split_dataset,
    split_issues,
    write_sft_records,
)
from tests.factories import make_record
from tests.generation import install_project

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def records_of(groups: int, per_group: int = 3, seed: int = 0) -> list[SFTRecord]:
    rng = random.Random(seed)
    names = [f"load:G{rng.randrange(10**9):09d}" for _ in range(groups)]
    return [make_record(id=f"r-{g}-{i}", group_id=name, split=None)
            for g, name in enumerate(names) for i in range(per_group)]  # fmt: skip


def test_no_group_in_two_splits() -> None:
    assigned = assign_splits(records_of(1000), RATIOS, seed=42)
    splits_of: dict[str, set[str | None]] = {}
    for record in assigned:
        splits_of.setdefault(record.group_id, set()).add(record.split)
    assert len(splits_of) == 1000
    assert all(len(found) == 1 for found in splits_of.values())
    assert split_issues({"all": assigned}) != []  # split names differ from the file name
    grouped: dict[str, list[SFTRecord]] = {}
    for record in assigned:
        grouped.setdefault(str(record.split), []).append(record)
    assert split_issues(grouped) == []


def test_ratios_approximately_respected() -> None:
    assigned = assign_splits(records_of(1000, per_group=1), RATIOS, seed=42)
    counts = Counter(record.split for record in assigned)
    assert 760 <= counts["train"] <= 840
    assert 70 <= counts["val"] <= 130 and 70 <= counts["test"] <= 130


def test_split_deterministic_and_order_independent() -> None:
    records = records_of(200)
    first = assign_splits(records, RATIOS, seed=42)
    assert first == assign_splits(records, RATIOS, seed=42)
    shuffled = list(records)
    random.Random(1).shuffle(shuffled)
    by_id = {r.id: r.split for r in assign_splits(shuffled, RATIOS, seed=42)}
    assert by_id == {r.id: r.split for r in first}
    assert [r.split for r in first] != [r.split for r in assign_splits(records, RATIOS, seed=7)]


def test_holdout_groups_go_to_test_ood() -> None:
    records = records_of(20)
    held = {records[0].group_id}
    assigned = assign_splits(records, RATIOS, seed=42, holdout=held)
    assert {r.split for r in assigned if r.group_id in held} == {"test_ood"}
    assert "test_ood" not in {r.split for r in assigned if r.group_id not in held}


def test_group_hash_splitter_is_registered_and_validated() -> None:
    assert SPLITTERS.get("group_hash") is GroupHashSplitter
    splitter = GroupHashSplitter(GroupHashSplitter.Config(seed=1, ratios={"train": 1.0}))
    assert isinstance(splitter, Splitter)
    assert {r.split for r in splitter.assign(records_of(5))} == {"train"}
    with pytest.raises(ValidationError, match="sum to 1"):
        GroupHashSplitter.Config(seed=1, ratios={"train": 0.5})


def test_split_issues_report_leaks_and_mismatches() -> None:
    record = make_record()
    issues = split_issues({"train": [record], "val": [record.model_copy(update={"id": "x"})]})
    assert sorted((i.record_id, i.code) for i in issues) == [
        ("qf-train-LOAD00001-0", "GROUP_LEAK"), ("x", "GROUP_LEAK"), ("x", "SPLIT_MISMATCH"),
    ]  # fmt: skip


# --- CLI -----------------------------------------------------------------------------------


def test_cli_split_checks_generated_dataset(
    fake_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_project(fake_project)
    assert main(["data", "build"]) == 0
    assert main(["data", "split"]) == 0
    assert "OK: splits already assigned" in capsys.readouterr().out


def test_cli_split_assigns_unsplit_records(fake_project: Path) -> None:
    (fake_project / "configs" / "data").mkdir(parents=True)
    split_config = Path(__file__).parents[1] / "configs" / "data" / "split.yaml"
    (fake_project / "configs/data/split.yaml").write_text(split_config.read_text())
    source = fake_project / "data" / "manual" / "requests.jsonl"
    source.parent.mkdir(parents=True)
    input_ref = write_sft_records(records_of(30), source, run_id="run-m", parents=[],
                                  root=fake_project)  # fmt: skip
    assert main(["data", "split", "--input", "data/manual/requests.jsonl",
                 "--out-dir", "data/manual/split"]) == 0  # fmt: skip
    out = fake_project / "data" / "manual" / "split"
    total = 0
    for path in sorted(out.glob("*.jsonl")):
        records, issues = read_sft_records(path)
        assert issues == [] and {r.split for r in records} <= {path.stem}  # test_ood: empty
        assert load_artifact_manifest(path).ref.parents == (input_ref.sha256,)
        total += len(records)
    assert total == 90
    manifest = read_manifest(
        fake_project / "runs" / load_artifact_manifest(path).ref.producer_run_id
    )
    assert manifest.config["splitter"]["name"] == "group_hash"
    with pytest.raises(QFError, match="already have a split"):
        split_dataset(GroupHashSplitter(GroupHashSplitter.Config(seed=1, ratios={"train": 1.0})),
                      path, root=fake_project, out_dir=out, config={})  # fmt: skip
    assert main(["data", "split", "--input", "data/manual/requests.jsonl"]) == 1


def test_hand_made_records_split_then_validate(
    fake_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: records without splits -> `qf data split --input` -> `qf validate-data`."""
    install_project(fake_project)
    assert main(["data", "build"]) == 0
    built = fake_project / "data" / "processed" / "generated_v1"
    unsplit = [
        SFTRecord.model_validate({**record.model_dump(), "split": None})
        for name in ("train", "val", "test")  # no T7/T8: those belong to test_ood only
        for record in read_sft_records(built / f"{name}.jsonl")[0]
    ]
    source = fake_project / "data" / "manual" / "requests.jsonl"
    source.parent.mkdir(parents=True)
    write_sft_records(unsplit, source, run_id="run-m", parents=[], root=fake_project)
    (fake_project / "configs/data/split.yaml").write_text(
        (Path(__file__).parents[1] / "configs/data/split.yaml").read_text()
    )
    assert main(["data", "split", "--input", "data/manual/requests.jsonl",
                 "--out-dir", "data/manual/split"]) == 0  # fmt: skip
    out = fake_project / "data" / "manual" / "split"
    assert sorted(p.stem for p in out.glob("*.jsonl")) == ["test", "test_ood", "train", "val"]
    assert read_sft_records(out / "test_ood.jsonl") == ([], [])
    capsys.readouterr()
    assert main(["validate-data", "--data-dir", "data/manual/split"]) == 0
    assert "OK: 0 errors" in capsys.readouterr().out
