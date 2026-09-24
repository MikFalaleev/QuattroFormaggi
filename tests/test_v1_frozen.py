"""Version 1 is frozen and byte-reproducible (D-080, D-086): card_v1 data, prompt, benchmark
and the step 9 acceptance runs of the eval (sub-step V6 shares the eval code with card_v2).

`card_v2` shares the generator, the vocabulary and the record checks with version 1. These
pinned hashes fail as soon as shared code changes what version 1 produces or accepts. The
constants are the hashes of the committed artifacts (manifests of the `qf data facts` and
`qf data build` runs, bench_v1 of step 8); they are pinned here, not read from the manifests,
because a re-run rewrites the manifests.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable
from pathlib import Path

import pytest
from pydantic import BaseModel

from qf.cli.main import main
from qf.common import sha256_file, sha256_text
from qf.contracts import SFTRecord
from qf.data import generate_records
from qf.domain import check_record, system_prompt_hash
from tests import fake_eval
from tests.conftest import REPO_ROOT
from tests.generation import REAL_RAW, facts_of, repo_config
from tests.test_generate_determinism import _digest

PROMPT_V1 = "c3188b85653eef9f7e0c810a9c30e8cc046ff897291f81f53e0d823092e6c75f"
SYNTHETIC_GENERATION = "e318414318bee921eee88d2260f9e148db29b5c2bcb7c0a8b962df7a0ebf8835"
FIXTURE_FACTS = "1f09d8654e309712ac931f92f281cb9109747644cdfe5eb26089a1b0bec2d979"
REAL_FACTS = "d8e8a92e5163fe51be46ee75a41ec5ad3df26619de5a0248e5c77a08eb59348a"
REAL_SOURCE_PREFIX = "yogape/logistics-operations@54e7d1d1a437"
GENERATED_V1 = {
    "train": "c8a134b3e47b97d15a9c52eb769458ed26f78847821f9a39d6c4484f789b2e7a",
    "val": "a8b1c8b3d3b508ca996fa5e90a53f9fdaf0e50058b6f888fb1cb94ec4d8b98c8",
    "test": "e90f6d53fdcc20696443356f0b0440b70dff606754a20b7ca6f2a37910bda9e2",
    "test_ood": "4c62d46f848f2edab9ead838c3f8a3c05e039d105fd5d37055661a11074355e8",
    "smoke": "79d72282225f68b5b755e0f19837be2a2587ff94c234ee6a79b5b1f0a5fc4f53",
}
BENCH_V1 = REPO_ROOT / "data" / "splits" / "bench_v1.jsonl"
BENCH_V1_SHA256 = "113a59994cbc164bb2df97b8a6058cf2e437abe2e86ee270be33e2c78fda2487"
GENERATED_DIR = REPO_ROOT / "data" / "processed" / "generated_v1"
# The step 9 acceptance runs on bench_v1 (20260923-173711-eval-42d036 and
# 20260923-173712-eval-bc9c83): scores.jsonl, metrics.json and report.md without its run-id line
# (predictions differ only in measured latency).
FAKE_RUNS_V1 = {
    "fake_bench_v1": (
        "e7248b1f6a82db72d799bc3159bc2dfa32560802d3ac05be4c451a90307a2f0d",
        "eaa874856f2a6d9a593c9a71f44386e1fb5b902bc9c5cb5a0aac43706b22e3c2",
        "adeb6682fc794eccef8ae86f20c2f06ab9a69fe74b92144f0a82a0088116a539",
    ),
    "fake_bench_v1_gold": (
        "dbceb5a2bb6753a8d4902e7582afe84ecbcd71009eae3b482207a067dc28b080",
        "92b935cc517eaea90cbf4531a3111a43577abc30fdb8a6b0f9e9a9e40c2368fb",
        "cd41ec3de6f487d687b52eef56aafc5966eebdeec21760d2efba39c885ebd5ed",
    ),
}


def jsonl_sha256(items: Iterable[BaseModel]) -> str:
    """sha256 of the JSONL file `items` are written as (`save_facts`, `write_sft_records`)."""
    text = "".join(item.model_dump_json() + "\n" for item in items)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_prompt_v1_unchanged() -> None:
    assert system_prompt_hash("system_extract_v1") == PROMPT_V1


def test_generation_v1_on_synthetic_facts_unchanged() -> None:
    assert _digest() == SYNTHETIC_GENERATION


def test_facts_v1_of_the_fixture_unchanged() -> None:
    assert jsonl_sha256(facts_of()) == FIXTURE_FACTS


@pytest.mark.slow
@pytest.mark.skipif(not REAL_RAW.exists(), reason="raw dataset not fetched")
def test_real_facts_and_generated_v1_unchanged() -> None:
    """`qf data facts` + `qf data build` with the v1 configs give the committed bytes."""
    facts = sorted(facts_of(REAL_RAW), key=lambda fact: fact.load_id)
    assert jsonl_sha256(facts) == REAL_FACTS
    cfg = repo_config()
    records, _ = generate_records(facts, cfg, source_prefix=REAL_SOURCE_PREFIX)
    outputs = {**records, "smoke": records["train"][: cfg.pool.smoke]}
    assert {name: jsonl_sha256(items) for name, items in outputs.items()} == GENERATED_V1


@pytest.mark.slow
@pytest.mark.skipif(not BENCH_V1.exists() or not GENERATED_DIR.exists(),
                    reason="run `qf data build` and freeze bench_v1 first")  # fmt: skip
def test_v1_records_still_pass_record_checks() -> None:
    """The record checks (shared with card_v2) still accept every committed v1 record."""
    assert sha256_file(BENCH_V1) == BENCH_V1_SHA256
    files = [BENCH_V1, *(GENERATED_DIR / f"{name}.jsonl" for name in GENERATED_V1)]
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = SFTRecord.model_validate_json(line)
            assert check_record(record) == [], (path.name, record.id)


@pytest.mark.slow
@pytest.mark.skipif(not BENCH_V1.exists(), reason="freeze bench_v1 first (step 8)")
def test_step9_acceptance_runs_unchanged(fake_project: Path) -> None:
    """The fake runs of step 9 give the same scores, metrics and report as when committed."""
    splits = fake_project / "data" / "splits"
    splits.mkdir(parents=True)
    for suffix in (".jsonl", ".sha256", ".jsonl.manifest.json"):
        shutil.copy(BENCH_V1.with_name(f"bench_v1{suffix}"), splits)
    fake_eval.build(fake_project)
    for name, (scores, metrics, report) in FAKE_RUNS_V1.items():
        runs = fake_project / "runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        config = fake_project / fake_eval.DEFAULT_OUT / f"{name}.yaml"
        assert main(["eval", "run", "--config", str(config)]) == 0
        (run,) = set(runs.iterdir()) - before
        assert sha256_file(run / "scores.jsonl") == scores, name
        assert sha256_file(run / "metrics.json") == metrics, name
        text = (run / "report.md").read_text(encoding="utf-8")
        kept = "".join(line for line in text.splitlines(keepends=True)
                       if not line.startswith("- Прогон:"))  # fmt: skip
        assert sha256_text(kept) == report, name
