"""`qf validate-data`, `qf data split` and `qf data report` on card_v2 data (sub-step V4)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from qf.cli.main import main
from qf.common import load_artifact_manifest, read_artifact, write_artifact
from qf.contracts import supported_versions
from qf.data import ValidationReport, run_validation, validate_dataset
from tests.conftest import REPO_ROOT
from tests.generation import CONDITIONS_TABLE, install_project, repo_config_v2

DATA = Path("data/processed/generated_v2")
SPLITS = ("train", "val", "test", "test_ood", "smoke")


@pytest.fixture
def project(fake_project: Path) -> Path:
    """generated_v2 of the 10-load fixture, built with `qf data build --config`."""
    install_project(fake_project)
    configs = fake_project / "configs" / "data"
    shutil.copy(CONDITIONS_TABLE, configs / CONDITIONS_TABLE.name)
    pool = {"train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2}
    ood = {"holdout_route_count": 1, "holdout_families": ["T7", "T8"]}
    cfg = repo_config_v2(pool=pool, ood=ood, hard_case_floor={}, review_samples=0)
    path = configs / "generate_v2.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True))
    assert main(["data", "build", "--config", str(path)]) == 0
    return fake_project


def rows(project: Path, name: str) -> list[dict[str, Any]]:
    path = project / DATA / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rewrite(project: Path, name: str, data: list[dict[str, Any]]) -> None:
    """Replace a split file and its manifest (same parents: the lineage to the facts stays)."""
    path = project / DATA / f"{name}.jsonl"
    manifest = load_artifact_manifest(path, root=project)
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in data)
    path.write_text(text, encoding="utf-8")
    write_artifact(path, "sft_dataset", manifest.ref.schema_version, manifest.ref.producer_run_id,
                   manifest.ref.parents, extra=manifest.extra, root=project)  # fmt: skip


def validate(project: Path) -> ValidationReport:
    paths = {name: project / DATA / f"{name}.jsonl" for name in SPLITS}
    return run_validation(paths, None, root=project,
                          out_path=project / DATA / "validation_report.json").report  # fmt: skip


def test_card_v2_dataset_passes(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-data", "--data-dir", str(project / DATA)]) == 0
    assert "OK: 0 errors" in capsys.readouterr().out
    assert main(["data", "split", "--data-dir", str(project / DATA)]) == 0


def test_route_leak_is_found_through_facts_v2(project: Path) -> None:
    """The route of a card_v2 record is looked up in load_facts_v2, its nearest facts."""
    ood = rows(project, "test_ood")
    moved = next(r for r in ood if r["variant"]["ood_reason"] == "route")
    rewrite(project, "test_ood", [r for r in ood if r is not moved])
    moved = {**moved, "id": moved["id"].replace("test_ood", "train"), "split": "train",
             "variant": {**moved["variant"], "ood_reason": None}}  # fmt: skip
    rewrite(project, "train", [*rows(project, "train"), moved])
    codes = {(issue.record_id, issue.code) for issue in validate(project).errors}
    assert (moved["id"], "OOD_ROUTE_LEAK") in codes


def test_card_v2_answer_errors_are_found(project: Path) -> None:
    data = rows(project, "val")
    answer = json.loads(data[0]["messages"][2]["content"])
    answer["missing_fields"] = ["pieces"]  # not what the card_v2 rule says
    data[0]["messages"][2]["content"] = json.dumps(answer, ensure_ascii=False,
                                                   separators=(",", ":"))  # fmt: skip
    rewrite(project, "val", data)
    codes = {issue.code for issue in validate(project).errors}
    assert "TARGET_INCONSISTENT" in codes


def test_length_report_counts_conditions(project: Path) -> None:
    assert main(["data", "report", "--data-dir", str(project / DATA)]) == 0
    report = json.loads((project / DATA / "length_report.json").read_text(encoding="utf-8"))
    composition = report["composition"]["train"]
    assert "special_conditions" in composition
    assert sum(composition["special_conditions"].values()) >= report["records"]["train"]


REAL = Path("data/processed/generated_v2")
BUILD_FIRST = "run `qf data build --config configs/data/generate_v2.yaml` first"


@pytest.mark.slow
@pytest.mark.skipif(not (REPO_ROOT / REAL / "train.jsonl").exists(), reason=BUILD_FIRST)
def test_real_generated_v2_is_valid() -> None:
    """The committed generator's real output: no leak, no invalid record (read only)."""
    refs = {name: read_artifact(REPO_ROOT / REAL / f"{name}.jsonl", "sft_dataset",
                                supported_versions("sft_dataset"), root=REPO_ROOT)
            for name in SPLITS}  # fmt: skip
    report = validate_dataset(refs, None, root=REPO_ROOT)
    assert report.errors == [] and report.warnings == []
    assert report.datasets["train"]["records"] == 1500
