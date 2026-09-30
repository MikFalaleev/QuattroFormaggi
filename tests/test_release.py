"""Step 19 (D-129): the release manifest and `release_check` on a small complete fixture: a chain
raw dataset -> facts -> training data -> adapter -> merged model -> GGUF, two cards and the
specifications."""

from __future__ import annotations

from pathlib import Path

import pytest

from qf.cli.main import main
from qf.common import write_artifact
from qf.export import RELEASE_MANIFEST, build_release, release_check

CARD_EN = """# Quattro-Formaggi-12B-Logistics-v0.1

Hashes: adapter {adapter}, merged {merged}, gguf {gguf}.

| Gate | Measured | Verdict |
|---|---|---|
| P1 json >= 0.98 | 0.9889 | met |
| P2 key >= 0.95 | 0.9505 | met |
| P3 zero critical | 0 / 0 / 1 | not met (exception) |
| P3m at most 2 | 4 | not met (exception) |
| P4 +5 pp | +0.946 | met |
| P5 loss <= 2 pp | +0.009 | met |
| P6 above rules | 0.9505 / 0.741 | met |
"""
CARD_RU = CARD_EN.replace("| P", "| П").replace("Gate", "Порог")


def _write(path: Path, text: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def release(fake_project: Path) -> Path:
    root = fake_project
    raw = root / "data/raw/src"
    _write(raw / "a.csv", "raw")
    r = write_artifact(raw, "raw_dataset", "logistics_ops_csv_v1", "run-raw", root=root)
    facts = _write(root / "data/processed/facts.jsonl", "facts")
    f = write_artifact(facts, "load_facts", "load_facts_v1", "run-facts", [r.sha256], root=root)
    train = _write(root / "data/processed/train.jsonl", "train")
    t = write_artifact(train, "sft_dataset", "sft_record_v1", "run-gen", [f.sha256], root=root)
    adapter = root / "artifacts/adapters/a"
    _write(adapter / "adapter_model.safetensors", "lora")
    a = write_artifact(adapter, "lora_adapter", "peft_lora_v1", "run-train", [t.sha256], root=root)
    merged = root / "artifacts/merged/m"
    _write(merged / "model.safetensors", "merged")
    m = write_artifact(merged, "hf_model", "hf_safetensors_v1", "run-merge", [a.sha256], root=root)
    gguf = _write(root / "artifacts/gguf/m-Q6_K.gguf", "gguf")
    g = write_artifact(gguf, "gguf", "gguf_v3", "run-gguf", [m.sha256], root=root)
    cards = root / "docs/release"
    hashes = {"adapter": a.sha256[:12], "merged": m.sha256[:12], "gguf": g.sha256[:12]}
    _write(cards / "README.md", CARD_EN.format(**hashes))
    _write(cards / "README.ru.md", CARD_RU.format(**hashes))
    _write(root / "docs/EVAL_SPEC.md")
    _write(root / "docs/DATA_SPEC.md")
    out = root / "artifacts/release"
    build_release(
        out, release="Quattro-Formaggi-12B-Logistics-v0.1", license_id="Apache-2.0",
        base_model="mistralai/Mistral-Nemo-Instruct-2407", base_revision="04d8a905",
        artifacts={"adapter": adapter, "merged": merged, "gguf": gguf},
        card_sources={"README.md": cards / "README.md", "README.ru.md": cards / "README.ru.md"},
        docs=["docs/EVAL_SPEC.md", "docs/DATA_SPEC.md"], root=root,
    )  # fmt: skip
    return out


def test_release_check_passes_on_complete_fixture(release: Path) -> None:
    assert release_check(release) == []
    assert release_check(release, verify_hashes=False) == []


def test_release_check_missing_file(release: Path, fake_project: Path) -> None:
    (fake_project / "artifacts/gguf/m-Q6_K.gguf").unlink()
    problems = release_check(release)
    assert any("gguf" in p and "missing" in p for p in problems)
    (release / "README.ru.md").unlink()
    assert any("README.ru.md" in p and "missing" in p for p in release_check(release))


def test_release_check_placeholder_detected(release: Path) -> None:
    card = release / "README.md"
    card.write_text(card.read_text(encoding="utf-8") + "\nRun <run_id> to reproduce.\n",
                    encoding="utf-8")  # fmt: skip
    assert any("placeholder '<run_id>'" in p for p in release_check(release))


def test_release_check_hash_mismatch(release: Path, fake_project: Path) -> None:
    (fake_project / "artifacts/adapters/a/adapter_model.safetensors").write_text("tampered")
    assert any("adapter" in p and "hash mismatch" in p for p in release_check(release))
    # the cheap check compares manifests only and says nothing about the content
    assert release_check(release, verify_hashes=False) == []


def test_lineage_break_is_found(release: Path, fake_project: Path) -> None:
    (fake_project / "data/processed/train.jsonl").unlink()
    assert any("lineage" in p and "sft_dataset" in p for p in release_check(release))
    (fake_project / "data/raw/src/a.csv").write_text("changed")
    assert any("lineage" in p and "hash mismatch" in p for p in release_check(release))


def test_lineage_must_reach_the_raw_dataset(release: Path, fake_project: Path) -> None:
    write_artifact(fake_project / "artifacts/gguf/m-Q6_K.gguf", "gguf", "gguf_v3", "run-x",
                   root=fake_project)  # fmt: skip
    problems = release_check(release, verify_hashes=False)
    assert any("no 'lora_adapter' ancestor" in p for p in problems)
    assert any("no 'raw_dataset' ancestor" in p for p in problems)


def test_card_must_name_every_artifact_hash(release: Path) -> None:
    card = release / "README.md"
    card.write_text(card.read_text(encoding="utf-8").replace("Hashes:", "Hashes (none):", 1)
                    .split("Hashes")[0] + "\n" + "\n".join(
                        line for line in card.read_text(encoding="utf-8").splitlines()
                        if line.startswith("|")), encoding="utf-8")  # fmt: skip
    assert any("sha256 of 'merged'" in p for p in release_check(release))


def test_gates_table_needs_actual_numbers(release: Path) -> None:
    card = release / "README.md"
    text = card.read_text(encoding="utf-8")
    text = text.replace("| P2 key >= 0.95 | 0.9505 | met |", "| P2 key >= 0.95 | TBD | met |")
    text = text.replace("| P6 above rules | 0.9505 / 0.741 | met |", "")
    card.write_text(text, encoding="utf-8")
    problems = release_check(release)
    assert any("gate 2 holds no actual number" in p for p in problems)
    assert any("no row for gate 6" in p for p in problems)


def test_missing_specification_and_manifest(release: Path, fake_project: Path) -> None:
    (fake_project / "docs/DATA_SPEC.md").unlink()
    assert any("DATA_SPEC.md" in p for p in release_check(release))
    (release / RELEASE_MANIFEST).write_text("{}", encoding="utf-8")
    assert any("invalid" in p for p in release_check(release))
    assert release_check(release / "nowhere")[0].startswith("missing or unreadable")


def test_cli_release_check(release: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["release", "check"]) == 0
    assert "complete (hashes verified)" in capsys.readouterr().out
    assert main(["release", "check", "--skip-hashes"]) == 0
    assert "content hashes NOT verified" in capsys.readouterr().out
    (release / "README.md").unlink()
    assert main(["release", "check"]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: README.md" in out and "NOT complete" in out


def test_cli_release_build_from_config(release: Path, fake_project: Path,
                                       capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    config = fake_project / "configs/release/r.yaml"
    _write(
        config,
        (
            "release: Quattro-Formaggi-12B-Logistics-v0.1\nlicense: Apache-2.0\n"
            "base_model: mistralai/Mistral-Nemo-Instruct-2407\nbase_revision: 04d8a905\n"
            "release_dir: artifacts/release2\n"
            "artifacts:\n  adapter: artifacts/adapters/a\n  merged: artifacts/merged/m\n"
            "  gguf: artifacts/gguf/m-Q6_K.gguf\n"
            "card_sources:\n  README.md: docs/release/README.md\n"
            "  README.ru.md: docs/release/README.ru.md\n"
            "docs: [docs/EVAL_SPEC.md, docs/DATA_SPEC.md]\n"
        ),
    )
    assert main(["release", "build", "--config", "configs/release/r.yaml"]) == 0
    assert "3 artifacts, 2 cards" in capsys.readouterr().out
    assert main(["release", "check", "--release-dir", "artifacts/release2"]) == 0
