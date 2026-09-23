from __future__ import annotations

import csv
import hashlib
import importlib
import json
import shutil
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qf.cli import main
from qf.common import QFError, load_yaml_config, read_artifact, read_manifest
from qf.contracts import RawSource, supported_versions
from qf.data import (
    PROVENANCE_FILENAME,
    RAW_SCHEMA_VERSION,
    RAW_SOURCES,
    HFDatasetSource,
    SourceConfig,
    SourceFileConfig,
    fetch_raw_dataset,
    read_provenance,
    verify_provenance,
    verify_raw_dataset,
    write_provenance,
)
from tests.conftest import REPO_ROOT

fetch_module = importlib.import_module("qf.data.fetch")

REVISION = "54e7d1d1a437ac9d9b287d3ce3ad0edea6aa7a07"
FILES = (
    "loads.csv", "routes.csv", "customers.csv", "delivery_events.csv", "README.md",
    "DATABASE_SCHEMA.txt",
)  # fmt: skip
TARGET = Path("data/raw/logistics-operations") / REVISION


def _config(**overrides: Any) -> SourceConfig:
    data: dict[str, Any] = {
        "repo_id": "yogape/logistics-operations",
        "revision": REVISION,
        "license": "MIT",
        "files": FILES,
        "excluded_reason": {"drivers.csv": "personal-data-shaped fields"},
    }
    data.update(overrides)
    return SourceConfig(**data)


class FakeHub:
    """Stands in for hf_hub_download(local_dir=...), including the cache dir it creates."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on = fail_on

    def __call__(
        self, repo_id: str, filename: str, *, repo_type: str, revision: str, local_dir: Path
    ) -> str:
        self.calls.append(
            {"repo_id": repo_id, "filename": filename, "repo_type": repo_type, "revision": revision}
        )
        if filename == self.fail_on:
            raise ConnectionError("network down")
        cache = local_dir / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{filename}.metadata").write_text("etag")
        path = local_dir / filename
        path.write_text(f"content of {filename}\n", encoding="utf-8")
        return str(path)

    @property
    def filenames(self) -> list[str]:
        return [call["filename"] for call in self.calls]


def _source(root: Path, hub: FakeHub | None = None, **overrides: Any) -> HFDatasetSource:
    return HFDatasetSource(_config(**overrides), root=root, downloader=hub or FakeHub())


def _fetch(root: Path, hub: FakeHub | None = None) -> tuple[HFDatasetSource, Any]:
    source = _source(root, hub)
    return source, source.fetch(root / "data" / "raw", "run-1")


# --- configuration -------------------------------------------------------------------------


@pytest.mark.parametrize("revision", ["main", REVISION[:12], REVISION.upper(), REVISION + "0"])
def test_revision_must_be_full_sha(revision: str) -> None:
    with pytest.raises(ValidationError, match="full 40-character"):
        _config(revision=revision)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"files": ("loads.csv", "loads.csv")}, "unique"),
        ({"files": ("../secrets.csv",)}, "plain top-level"),
        ({"files": ("sub/loads.csv",)}, "plain top-level"),
        ({"files": (".DS_Store",)}, "cannot be part"),
        ({"files": ()}, "at least 1"),
        ({"files": ("loads.csv", "drivers.csv")}, "both fetched and excluded: drivers.csv"),
        ({"repo_id": "no-slash"}, "pattern"),
    ],
)
def test_config_rejects_bad_values(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**overrides)


def test_repository_source_config_is_valid_and_pinned() -> None:
    file_config = load_yaml_config(REPO_ROOT / "configs/data/source.yaml", SourceFileConfig)
    assert file_config.source.name == "hf_dataset"
    config = SourceConfig.model_validate(file_config.source.model_extra)
    assert config.revision == REVISION
    assert config.files == FILES
    assert config.license == "MIT"
    assert "drivers.csv" in config.excluded_reason


def test_hf_source_implements_port_and_is_registered(tmp_path: Path) -> None:
    assert isinstance(_source(tmp_path), RawSource)
    assert RAW_SOURCES.port is RawSource
    assert RAW_SOURCES.get("hf_dataset") is HFDatasetSource
    assert HFDatasetSource.Config is SourceConfig


# --- fetch ---------------------------------------------------------------------------------


def test_fetch_downloads_only_listed_files(fake_project: Path) -> None:
    hub = FakeHub()
    _fetch(fake_project, hub)
    assert hub.filenames == list(FILES)
    assert "drivers.csv" not in hub.filenames
    assert {(c["repo_type"], c["revision"]) for c in hub.calls} == {("dataset", REVISION)}


def test_fetch_creates_clean_verified_artifact(fake_project: Path) -> None:
    _, ref = _fetch(fake_project)
    target = fake_project / TARGET
    assert sorted(p.name for p in target.iterdir()) == sorted([*FILES, PROVENANCE_FILENAME])
    assert sorted(p.name for p in target.parent.iterdir()) == [
        REVISION,
        f"{REVISION}.manifest.json",
    ]
    assert (ref.kind, ref.schema_version, ref.producer_run_id) == (
        "raw_dataset", RAW_SCHEMA_VERSION, "run-1",
    )  # fmt: skip
    assert read_artifact(TARGET, "raw_dataset", supported_versions("raw_dataset")) == ref


def test_failed_download_leaves_nothing_behind(fake_project: Path) -> None:
    source = _source(fake_project, FakeHub(fail_on="delivery_events.csv"))
    with pytest.raises(QFError, match="delivery_events.csv .* failed: ConnectionError: network"):
        source.fetch(fake_project / "data" / "raw", "run-1")
    assert not (fake_project / TARGET).exists()
    assert list((fake_project / TARGET).parent.iterdir()) == []  # no staging leftovers


def test_download_without_a_file_is_an_error(fake_project: Path) -> None:
    source = HFDatasetSource(_config(), root=fake_project, downloader=lambda *a, **k: None)
    with pytest.raises(QFError, match="returned no file"):
        source.fetch(fake_project / "data" / "raw", "run-1")


def test_fetch_refuses_existing_target(fake_project: Path) -> None:
    source, _ = _fetch(fake_project)
    with pytest.raises(QFError, match="already exists"):
        source.fetch(fake_project / "data" / "raw", "run-2")


# --- provenance and verification -----------------------------------------------------------


def test_provenance_hashes(tmp_path: Path) -> None:
    for name in FILES:
        (tmp_path / name).write_text(f"data {name}", encoding="utf-8")
    path = write_provenance(tmp_path, _config())
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in FILES:
        raw = (tmp_path / name).read_bytes()
        assert data["files"][name] == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert (data["repo_id"], data["revision"], data["license"]) == (
        "yogape/logistics-operations", REVISION, "MIT",
    )  # fmt: skip
    assert data["excluded"] == {"drivers.csv": "personal-data-shaped fields"}
    assert read_provenance(tmp_path).fetched_at.tzinfo == UTC
    assert verify_provenance(tmp_path) == []


def test_write_provenance_requires_every_file(tmp_path: Path) -> None:
    with pytest.raises(QFError, match="loads.csv is missing"):
        write_provenance(tmp_path, _config())


def test_verify_detects_modified_file(fake_project: Path) -> None:
    source, ref = _fetch(fake_project)
    loads = fake_project / TARGET / "loads.csv"
    original = loads.read_bytes()
    loads.write_bytes(original[:-2] + b"X\n")  # same size, one byte changed
    problems = source.verify(ref)
    assert "loads.csv: sha256 differs from provenance" in problems
    assert any("content hash differs" in p for p in problems)
    loads.write_bytes(original + b"extra")
    assert any(p.startswith("loads.csv: size") for p in source.verify(ref))


def test_verify_detects_missing_and_unexpected_files(fake_project: Path) -> None:
    source, ref = _fetch(fake_project)
    target = fake_project / TARGET
    (target / "routes.csv").unlink()
    (target / "drivers.csv").write_text("names, licences")
    (target / ".DS_Store").write_bytes(b"finder")  # OS junk is ignored everywhere
    problems = source.verify(ref)
    assert "routes.csv: missing" in problems
    assert "drivers.csv: unexpected file" in problems
    assert not any(".DS_Store" in p for p in problems)


def test_verify_detects_broken_provenance(fake_project: Path) -> None:
    source, ref = _fetch(fake_project)
    provenance = fake_project / TARGET / PROVENANCE_FILENAME
    provenance.write_text("{}", encoding="utf-8")
    assert any("invalid provenance.json" in p for p in source.verify(ref))
    provenance.unlink()
    assert any("missing or unreadable" in p for p in source.verify(ref))
    shutil.rmtree(fake_project / TARGET)
    assert source.verify(ref) == [f"{TARGET.as_posix()}: directory is missing"]


def test_verify_detects_config_drift(fake_project: Path) -> None:
    _, ref = _fetch(fake_project)
    drifted = _source(fake_project, files=FILES[:-1], license="Apache-2.0")
    problems = drifted.verify(ref)
    assert any(p.startswith("files: provenance has") for p in problems)
    assert "license: provenance has 'MIT', config has 'Apache-2.0'" in problems


# --- use case ------------------------------------------------------------------------------


def test_fetch_raw_dataset_is_idempotent_and_records_run(fake_project: Path) -> None:
    hub = FakeHub()
    source = _source(fake_project, hub)
    first = fetch_raw_dataset(source, root=fake_project, source_config={"name": "hf_dataset"})
    assert first.downloaded and first.run_dir is not None
    manifest = read_manifest(first.run_dir)
    assert (manifest.kind, manifest.status) == ("fetch", "completed")
    assert manifest.data_hashes == {TARGET.as_posix(): first.ref.sha256}
    assert manifest.config == {"name": "hf_dataset"}
    second = fetch_raw_dataset(source, root=fake_project, source_config={})
    assert (second.downloaded, second.run_dir, second.ref) == (False, None, first.ref)
    assert len(hub.calls) == len(FILES)  # nothing downloaded the second time
    assert len(list((fake_project / "runs").iterdir())) == 1


def test_fetch_raw_dataset_refuses_modified_dataset(fake_project: Path) -> None:
    hub = FakeHub()
    source = _source(fake_project, hub)
    fetch_raw_dataset(source, root=fake_project, source_config={})
    (fake_project / TARGET / "customers.csv").write_text("tampered")
    with pytest.raises(QFError, match="customers.csv: size"):
        fetch_raw_dataset(source, root=fake_project, source_config={})
    assert len(hub.calls) == len(FILES)


def test_verify_raw_dataset_before_fetch(fake_project: Path) -> None:
    with pytest.raises(QFError, match="run `qf data fetch` first"):
        verify_raw_dataset(_source(fake_project), root=fake_project)


# --- CLI -----------------------------------------------------------------------------------


@pytest.fixture
def cli_project(fake_project: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    (fake_project / "configs" / "data").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs/data/source.yaml", fake_project / "configs/data/source.yaml")
    hub = FakeHub()
    monkeypatch.setattr(fetch_module, "_hf_download", hub)
    return hub


def test_cli_fetch_then_verify(
    fake_project: Path, cli_project: FakeHub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "fetch", "--verify-only"]) == 1
    assert "run `qf data fetch` first" in capsys.readouterr().err
    assert main(["data", "fetch"]) == 0
    assert "Downloaded data/raw/logistics-operations/" in capsys.readouterr().out
    assert main(["data", "fetch"]) == 0
    assert "Already present and verified" in capsys.readouterr().out
    assert main(["data", "fetch", "--verify-only"]) == 0
    assert "matches its provenance" in capsys.readouterr().out
    assert cli_project.filenames == list(FILES)


def test_cli_reports_modified_files(
    fake_project: Path, cli_project: FakeHub, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "fetch"]) == 0
    (fake_project / TARGET / "loads.csv").write_text("tampered")
    capsys.readouterr()
    assert main(["data", "fetch", "--verify-only"]) == 1
    assert "loads.csv: size" in capsys.readouterr().err
    assert main(["data", "fetch"]) == 1
    assert "loads.csv: size" in capsys.readouterr().err


def test_cli_explicit_config_path(
    fake_project: Path, cli_project: FakeHub, capsys: pytest.CaptureFixture[str]
) -> None:
    moved = fake_project / "my_source.yaml"
    (fake_project / "configs/data/source.yaml").rename(moved)
    assert main(["data", "fetch"]) == 1
    assert "cannot read config" in capsys.readouterr().err
    assert main(["data", "fetch", "--config", str(moved)]) == 0


# --- real download (run only with the user's permission) -----------------------------------

EXPECTED_ROWS = {"loads.csv": 85_410, "routes.csv": 58, "customers.csv": 200}
EXPECTED_ROWS["delivery_events.csv"] = 170_820


def _data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1  # minus the header


@pytest.mark.network
def test_fetch_real(fake_project: Path) -> None:
    file_config = load_yaml_config(REPO_ROOT / "configs/data/source.yaml", SourceFileConfig)
    source = HFDatasetSource(
        SourceConfig.model_validate(file_config.source.model_extra), root=fake_project
    )
    result = fetch_raw_dataset(source, root=fake_project, source_config={})
    assert result.downloaded
    for name, rows in EXPECTED_ROWS.items():
        assert _data_rows(fake_project / TARGET / name) == rows, name
    assert verify_raw_dataset(source, root=fake_project)[1] == []
