"""The harness with a batch backend and with a whole split instead of the benchmark (step 14,
D-120): the same predictions as one request at a time, failures per case, resume checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from qf.backends import FakeBackend
from qf.common import ArtifactRef, QFError, write_artifact
from qf.contracts import BatchGenerationBackend, GenerationRequest, GenerationResult, SFTRecord
from qf.eval import BenchRef, load_bench
from tests.test_harness import bench_records, config, gold_answers, install_bench, run

SPLIT = Path("data/splits/test.jsonl")


class FakeBatch(FakeBackend):
    """Prepared answers, `size` requests per call; `fail_call` makes that call raise."""

    def __init__(self, answers: dict[str, str], size: int, fail_call: int | None = None) -> None:
        super().__init__(FakeBackend.from_answers(answers).config)
        self.size, self.fail_call = size, fail_call
        self.calls: list[int] = []

    @property
    def batch_size(self) -> int:
        return self.size

    def generate_batch(self, reqs: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self.calls.append(len(reqs))
        if self.fail_call == len(self.calls):
            raise RuntimeError("out of memory")
        return [self.generate(r) for r in reqs]


@pytest.fixture
def bench(fake_project: Path) -> tuple[ArtifactRef, list[SFTRecord]]:
    records = bench_records()
    return install_bench(fake_project, records), records


def test_fake_batch_is_a_batch_backend() -> None:
    assert isinstance(FakeBatch({}, 2), BatchGenerationBackend)
    assert not isinstance(FakeBackend(FakeBackend.Config()), BatchGenerationBackend)


def test_batches_give_the_same_predictions(fake_project: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    single = run(fake_project, bench, FakeBackend.from_answers(answers))
    backend = FakeBatch(answers, size=3)
    batched = run(fake_project, bench, backend)
    n = len(bench[1])
    assert backend.calls == [3] * (n // 3) + ([n % 3] if n % 3 else [])
    assert {k: v["output"] for k, v in batched.predictions.items()} == {
        k: v["output"] for k, v in single.predictions.items()
    }
    assert batched.table["overall"] == single.table["overall"]
    assert batched.state["batch_size"] == 3 and "batch_size" not in single.state


def test_a_failing_batch_fails_each_of_its_cases(fake_project: Path, bench: Any) -> None:
    result = run(fake_project, bench, FakeBatch(gold_answers(bench[1]), size=3, fail_call=2))
    errors = [result.predictions[r.id]["error"] for r in bench[1]]
    assert errors[:3] == [None] * 3
    assert all(e == "backend raised RuntimeError: out of memory" for e in errors[3:6])
    assert all(e is None for e in errors[6:])


def test_a_wrong_number_of_results_is_an_error(fake_project: Path, bench: Any) -> None:
    class Short(FakeBatch):
        def generate_batch(self, reqs: Sequence[GenerationRequest]) -> list[GenerationResult]:
            return super().generate_batch(reqs)[:-1]

    result = run(fake_project, bench, Short(gold_answers(bench[1]), size=2))
    assert all(p["error"] == "backend returned 1 results for 2 requests"
               for p in result.predictions.values())  # fmt: skip


def test_resume_refuses_another_batch_size(fake_project: Path, bench: Any) -> None:
    answers = gold_answers(bench[1])
    first = run(fake_project, bench, FakeBatch(answers, size=3))
    with pytest.raises(QFError, match="batch_size changed"):
        run(fake_project, bench, FakeBatch(answers, size=2), resume=first.run_dir)


def install_split(root: Path, records: list[SFTRecord]) -> ArtifactRef:
    path = root / SPLIT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    return write_artifact(path, "sft_dataset", "sft_record_v1", "run-split", root=root)


def test_a_whole_split_can_be_evaluated(fake_project: Path) -> None:
    records = bench_records()
    ref = install_split(fake_project, records)
    cfg = config(ref, bench=BenchRef(path=SPLIT, sha256=ref.sha256, kind="sft_dataset"))
    loaded_ref, loaded = load_bench(cfg, fake_project)
    assert loaded_ref.sha256 == ref.sha256 and [r.id for r in loaded] == [r.id for r in records]
    with pytest.raises(QFError, match="expected artifact kind 'benchmark'"):
        load_bench(config(ref, bench=BenchRef(path=SPLIT, sha256=ref.sha256)), fake_project)
    with pytest.raises(QFError, match="differs from the config"):
        load_bench(config(ref, bench=BenchRef(path=SPLIT, sha256="0" * 64, kind="sft_dataset")),
                   fake_project)  # fmt: skip
