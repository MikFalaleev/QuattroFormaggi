"""Splits by group (plan step 7).

`GroupHashSplitter` (`group_hash`) assigns splits to records that come without one, e.g. future
hand-written or real requests: every record of a group lands in the same split, decided by a
stable hash of the seed and the group_id. generated_v1 comes with splits set by the generator;
`split_issues` checks any set of split files.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import Field, field_validator

from qf.common import (
    ArtifactRef,
    ComponentConfig,
    DataValidationError,
    Issue,
    QFError,
    StrictConfig,
    read_artifact,
    start_run,
)
from qf.contracts import SFTRecord, SplitName, Splitter, supported_versions
from qf.data.registries import SPLITTERS
from qf.data.sft_io import SPLIT_FILES, read_sft_records, write_sft_records

__all__ = [
    "GroupHashSplitter",
    "SplitFileConfig",
    "SplitResult",
    "assign_splits",
    "group_fraction",
    "split_dataset",
    "split_issues",
]

_ORDER: Final[tuple[SplitName, ...]] = ("train", "val", "test", "test_ood", "bench")


def group_fraction(seed: int, group_id: str) -> float:
    """A stable number in [0, 1) for a group: the first 8 bytes of sha256("<seed>:<group_id>")."""
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(
    records: Sequence[SFTRecord],
    ratios: Mapping[SplitName, float],
    seed: int,
    holdout: Collection[str] = (),
) -> list[SFTRecord]:
    """Records with `split` set: groups in `holdout` go to test_ood, the others by their hash
    fraction into consecutive ranges of `ratios` (taken in the order train, val, test, ...)."""
    bounds: list[tuple[float, SplitName]] = []
    total = 0.0
    for name in _ORDER:
        if name in ratios:
            total += ratios[name]
            bounds.append((total, name))
    assigned = []
    for record in records:
        if record.group_id in holdout:
            split: SplitName = "test_ood"
        else:
            fraction = group_fraction(seed, record.group_id) * total
            split = next((name for bound, name in bounds if fraction < bound), bounds[-1][1])
        assigned.append(SFTRecord.model_validate({**record.model_dump(), "split": split}))
    return assigned


@SPLITTERS.register("group_hash")
class GroupHashSplitter:
    """`Splitter` by a stable hash of the group (plan step 7)."""

    class Config(StrictConfig):
        seed: int
        ratios: dict[SplitName, float] = Field(min_length=1)
        holdout_groups: list[str] = Field(default_factory=list)

        @field_validator("ratios")
        @classmethod
        def _shares(cls, ratios: dict[SplitName, float]) -> dict[SplitName, float]:
            if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
                raise ValueError("ratios must be non-negative and sum to 1")
            return ratios

    def __init__(self, config: Config) -> None:
        self.config = config

    def assign(self, records: Sequence[SFTRecord]) -> list[SFTRecord]:
        return assign_splits(records, self.config.ratios, self.config.seed,
                             set(self.config.holdout_groups))  # fmt: skip


def split_issues(datasets: Mapping[str, Sequence[SFTRecord]]) -> list[Issue]:
    """GROUP_LEAK: a group in more than one split file; SPLIT_MISMATCH: a record whose `split`
    is not the name of its file. Pass split files only (not smoke)."""
    issues = []
    splits_of: dict[str, set[str]] = defaultdict(set)
    for name, records in datasets.items():
        for record in records:
            splits_of[record.group_id].add(name)
            if record.split != name:
                issues.append(Issue(record.id, name, "SPLIT_MISMATCH",
                                    f"split is {record.split!r}, file is {name!r}"))  # fmt: skip
    for name, records in datasets.items():
        for record in records:
            found = splits_of[record.group_id]
            if len(found) > 1:
                issues.append(Issue(record.id, name, "GROUP_LEAK",
                                    f"group {record.group_id} is in {sorted(found)}"))  # fmt: skip
    return issues


class SplitFileConfig(StrictConfig):
    """`configs/data/split.yaml`: which splitter assigns splits, with its parameters."""

    splitter: ComponentConfig


@dataclass(frozen=True)
class SplitResult:
    refs: dict[str, ArtifactRef]
    counts: dict[str, int]
    run_dir: Path


def split_dataset(
    splitter: Splitter, input_path: Path, *, root: Path, out_dir: Path, config: Mapping[str, Any]
) -> SplitResult:
    """Assign splits to an `sft_dataset` whose records have none; one output file per split,
    each with the input as its parent. train, val, test and test_ood are always written (empty
    if no group went there), so `qf validate-data` can check the result."""
    run = start_run("split", root)
    source = read_artifact(input_path, "sft_dataset", supported_versions("sft_dataset"), root=root)
    records, issues = read_sft_records(root / source.path)
    if issues:
        raise DataValidationError(issues)
    already = [record.id for record in records if record.split is not None]
    if already:
        raise QFError(f"{source.path}: records already have a split, e.g. {already[0]}")
    by_split: dict[str, list[SFTRecord]] = defaultdict(list)
    for record in splitter.assign(records):
        by_split[str(record.split)].append(record)
    refs = {
        name: write_sft_records(
            sorted(by_split[name], key=lambda r: r.id),
            out_dir / f"{name}.jsonl",
            run_id=run.run_id,
            parents=[source.sha256],
            root=root,
            extra={"split": name, "count": len(by_split[name])},
        )  # fmt: skip
        for name in sorted({*SPLIT_FILES, *by_split}, key=_ORDER.index)
    }
    counts = {name: len(by_split[name]) for name in refs}
    run.finish(
        config=dict(config),
        data_hashes={source.path.as_posix(): source.sha256,
                     **{ref.path.as_posix(): ref.sha256 for ref in refs.values()}},
        metrics={"counts": counts},
    )  # fmt: skip
    return SplitResult(refs=refs, counts=counts, run_dir=run.run_dir)
