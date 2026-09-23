from __future__ import annotations

import json
from pathlib import Path

import pytest

from qf.cli import main
from qf.common import load_artifact_manifest
from qf.data import length_report, length_stats
from qf.domain import serialize_target
from tests.factories import make_card, make_record, make_target
from tests.generation import install_project


def test_length_stats_nearest_rank() -> None:
    assert length_stats(list(range(1, 21))) == {"min": 1, "p50": 10, "p95": 19, "max": 20}
    assert length_stats([7]) == {"min": 7, "p50": 7, "p95": 7, "max": 7}
    assert length_stats([]) == {"min": 0, "p50": 0, "p95": 0, "max": 0}


def test_length_report_counts_and_lengths() -> None:
    reefer = serialize_target(make_target(make_card(equipment_type="reefer", pieces=None)))
    records = [make_record(), make_record(answer=reefer, id="b")]
    report = length_report({"train": records}, count_tokens=lambda text: len(text.split()))
    assert report["records"] == {"train": 2}
    composition = report["composition"]["train"]
    assert composition["equipment_type"] == {"dry_van": 1, "reefer": 1}
    assert composition["missing_fields"] == {"pieces": 1, "temperature_c": 1}
    assert composition["hard_case"] == {"clean": 2}
    user = len(records[0].messages[1].content)
    assert report["chars"]["train"]["user"] == {"min": user, "p50": user, "p95": user, "max": user}
    assert report["tokens"]["train"]["user"]["max"] == len(records[0].messages[1].content.split())
    assert length_report({"train": records})["tokens"] is None


def test_cli_report_writes_artifact(fake_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    install_project(fake_project)
    assert main(["data", "build"]) == 0
    assert main(["data", "report"]) == 0
    out = fake_project / "data" / "processed" / "generated_v1"
    stored = json.loads((out / "length_report.json").read_text(encoding="utf-8"))
    assert stored["records"] == {"train": 3, "val": 1, "test": 1, "test_ood": 3, "smoke": 2}
    assert (out / "length_report.md").read_text(encoding="utf-8").startswith("# Dataset length")
    manifest = load_artifact_manifest(out / "length_report.json")
    assert (manifest.ref.kind, manifest.ref.schema_version) == ("metrics", "length_report_v1")
    assert main(["data", "report", "--tokenizer", "tokenizer/"]) == 2
    assert "Stage 11 is not implemented yet" in capsys.readouterr().err
