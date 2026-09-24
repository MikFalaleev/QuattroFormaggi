"""The frozen benchmark (plan step 8): candidates, human review, freeze, verification.

Flow: `select_candidates` picks, per slice, the target number plus a small reserve from the
verified test and test_ood files (seeded, stratified); a human marks each candidate ok / fix /
drop in a CSV; `freeze` takes, per slice, the first ok records in the seeded priority order,
adds hand-written cases, writes `<version>.jsonl` + `.sha256` once and forever, and writes
test/test_ood without the benchmark records next to it. `generated_v1` is never changed.

A benchmark has one answer schema (`schema_version`): bench_v1 is card_v1 from generated_v1
into data/splits/, bench_v2 is card_v2 from generated_v2 into data/splits_v2/ (sub-step V5).
A card_v2 slice may also require a special condition or an equipment type in the gold card.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from qf.common import (
    ArtifactRef,
    DataValidationError,
    Issue,
    QFError,
    StrictConfig,
    atomic_write_text,
    load_artifact_manifest,
    read_artifact,
    sha256_file,
    start_run,
)
from qf.contracts import (
    CARD_SCHEMA_VERSION,
    ConditionKind,
    EquipmentTypeV2,
    SFTRecord,
    TargetSchemaVersion,
    supported_versions,
)
from qf.data.review import gold_lines
from qf.data.sft_io import DATASET_FILES, read_sft_records, write_sft_records
from qf.domain import check_record, get_target_schema, parse_target

__all__ = [
    "BENCH_SCHEMAS",
    "VERDICTS",
    "BenchmarkConfig",
    "Candidate",
    "ExportOutcome",
    "FreezeOutcome",
    "ReviewResult",
    "SliceConfig",
    "export_review",
    "freeze",
    "import_review",
    "import_review_stage",
    "read_verdicts",
    "select_candidates",
    "verify_benchmark",
]

BENCH_SCHEMAS: Final = ("card_v1", "card_v2")
"""Answer schemas this stage selects, reviews and freezes (card_v2 since sub-step V5)."""
VERDICTS: Final = ("ok", "fix", "drop")
CSV_COLUMNS: Final = ("n", "id", "slice", "role", "verdict", "comment")
Stratum = Literal["template_family", "language", "ood_reason"]


class SliceConfig(StrictConfig):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    count: int = Field(ge=1)
    reserve: int = Field(default=0, ge=0)  # extra candidates for review, used when some drop
    splits: list[Literal["test", "test_ood"]] = Field(min_length=1)
    kind: str  # "clean" or a hard case name
    stratify_by: list[Stratum] = Field(default_factory=list)
    # card_v2 only: the gold card must hold this condition kind / equipment type
    condition: ConditionKind | None = None
    equipment: EquipmentTypeV2 | None = None


class BenchmarkConfig(StrictConfig):
    """`configs/eval/benchmark_v1.yaml`. Paths are relative to the project root."""

    version: str = Field(pattern=r"^bench_v\d+$")
    seed: int
    schema_version: TargetSchemaVersion = CARD_SCHEMA_VERSION  # of every record, manual ones too
    source_dir: Path
    out_dir: Path
    review_csv: Path
    slices: list[SliceConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint_slices(self) -> BenchmarkConfig:
        names = [s.name for s in self.slices]
        keys = [
            (s.kind, split, s.condition, s.equipment) for s in self.slices for split in s.splits
        ]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ValueError("slice names and (kind, split, condition, equipment) must be unique")
        filtered = [s.name for s in self.slices if s.condition or s.equipment]
        if filtered and self.schema_version == CARD_SCHEMA_VERSION:
            raise ValueError(f"condition / equipment filters need card_v2 slices: {filtered}")
        return self

    @property
    def bench_path(self) -> Path:
        return self.out_dir / f"{self.version}.jsonl"

    @property
    def work_dir(self) -> Path:
        return self.out_dir / self.version


@dataclass(frozen=True)
class Candidate:
    slice: str
    role: Literal["main", "reserve"]
    record: SFTRecord  # still with its original split (test / test_ood)


def _kind(record: SFTRecord) -> str:
    return record.variant.hard_cases[0] if record.variant.hard_cases else "clean"


def _stratum(record: SFTRecord, keys: Sequence[Stratum]) -> tuple[str, ...]:
    values = {"template_family": record.template_family, "language": record.language,
              "ood_reason": str(record.variant.ood_reason)}  # fmt: skip
    return tuple(values[key] for key in keys)


def _systematic(
    pool: Sequence[SFTRecord], n: int, keys: Sequence[Stratum], rng: Random
) -> list[SFTRecord]:
    """n records spread over the strata in proportion to their sizes (random inside strata)."""
    ordered = list(pool)
    rng.shuffle(ordered)
    ordered.sort(key=lambda r: _stratum(r, keys))
    step = len(ordered) / n
    offset = rng.random() * step
    return [ordered[min(len(ordered) - 1, int(offset + i * step))] for i in range(n)]


def _matches(record: SFTRecord, spec: SliceConfig) -> bool:
    """The slice's condition and equipment filters (none for card_v1 slices)."""
    if spec.condition is None and spec.equipment is None:
        return True
    card = get_target_schema(record.schema_version).parse(record.messages[2].content).card
    kinds = {condition.kind for condition in getattr(card, "special_conditions", [])}
    return (spec.condition is None or spec.condition in kinds) and (
        spec.equipment is None or card.equipment_type == spec.equipment)  # fmt: skip


def select_candidates(
    datasets: Mapping[str, Sequence[SFTRecord]], cfg: BenchmarkConfig
) -> list[Candidate]:
    """Per slice: count + reserve (or all available) records in a seeded priority order; the
    first `count` are the main candidates, the rest the reserve. A slice that cannot reach its
    count is an error."""
    used: set[str] = set()
    candidates: list[Candidate] = []
    for spec in cfg.slices:
        pool = sorted(
            (r for split in spec.splits for r in datasets[split]
             if _kind(r) == spec.kind and r.id not in used and _matches(r, spec)),
            key=lambda r: r.id,
        )  # fmt: skip
        if len(pool) < spec.count:
            raise QFError(f"slice {spec.name}: needs {spec.count} records, {len(pool)} available")
        rng = Random(f"{cfg.seed}:{cfg.version}:{spec.name}")
        picked = _systematic(pool, min(spec.count + spec.reserve, len(pool)), spec.stratify_by, rng)
        rng.shuffle(picked)
        used |= {r.id for r in picked}
        candidates += [Candidate(spec.name, "main" if i < spec.count else "reserve", r)
                       for i, r in enumerate(picked)]  # fmt: skip
    return candidates


# --- review files --------------------------------------------------------------------------

_LABELS: Final = (
    ("shipper_name", "Отправитель"), ("cargo_category", "Категория"),
    ("equipment_type", "Тип кузова"), ("pieces", "Мест"), ("weight_total", "Вес (общий)"),
    ("weight_per_piece", "Вес одного места"), ("origin", "Откуда"), ("destination", "Куда"),
    ("pickup_date", "Загрузка"), ("delivery_date", "Доставка"), ("temperature_c", "Температура"),
)  # fmt: skip


def _readable(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict) and "unit" in value:
        return f"{value['value']} {value['unit']}"
    if isinstance(value, dict) and "city" in value:
        return f"{value['city']} ({value['region']})"
    return str(value)


def _gold_lines_v1(record: SFTRecord) -> list[str]:
    target = parse_target(record.messages[2].content).model_dump(mode="json")
    lines = [f"- {label}: {_readable(target['card'][key])}" for key, label in _LABELS]
    lines.append(f"- **Недостающие:** {', '.join(target['missing_fields']) or '—'}")
    conflicts = "; ".join(f"{c['field']}: {[_readable(v) for v in c['values']]}"
                          for c in target["conflicts"])  # fmt: skip
    lines.append(f"- **Конфликты:** {conflicts or '—'}")
    return lines


_GOLD_LINES: Final = {"card_v1": _gold_lines_v1, "card_v2": gold_lines}
"""The gold shown to the reviewer, per answer schema (card_v1 keeps its step 8 lines)."""


def render_review_markdown(candidates: Sequence[Candidate], version: str) -> str:
    lines = [f"# Проверка кандидатов {version}", "",
             "Для каждой заявки сравните текст с правильным ответом ниже и поставьте вердикт "
             "в CSV: `ok` — всё верно; `fix` — ответ или текст неверны (запись уйдёт на "
             "исправление генератора); `drop` — запись неудачная, но ошибки нет. Резервные "
             "заявки нужны, только если в срезе что-то отброшено.", ""]  # fmt: skip
    for n, candidate in enumerate(candidates, start=1):
        record = candidate.record
        role = "резерв" if candidate.role == "reserve" else "основная"
        user = record.messages[1].content
        lines += [f"## {n}. {record.id} — {candidate.slice} ({role}, {record.template_family})",
                  "", "```text", user, "```", "", *_GOLD_LINES[record.schema_version](record),
                  ""]  # fmt: skip
    return "\n".join(lines)


def _review_csv(candidates: Sequence[Candidate]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for n, c in enumerate(candidates, start=1):
        writer.writerow([n, c.record.id, c.slice, c.role, "", ""])
    return "﻿" + buffer.getvalue()  # BOM: Excel and Numbers then read UTF-8 correctly


def read_verdicts(path: Path) -> dict[str, tuple[str, str]]:
    """id -> (verdict, comment) from the review CSV (`,` or `;`, with or without BOM).
    Blank verdicts are kept as ""; any other value outside ok/fix/drop is an error."""
    text = path.read_text(encoding="utf-8-sig")
    delimiter = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    missing = [c for c in ("id", "verdict") if rows and c not in rows[0]]
    if missing or not rows:
        raise QFError(f"{path}: expected columns {', '.join(CSV_COLUMNS)}")
    verdicts, bad = {}, []
    for number, row in enumerate(rows, start=2):
        verdict = (row.get("verdict") or "").strip().lower()
        if verdict and verdict not in VERDICTS:
            bad.append(Issue(f"{path.name}:{number}", None, "BAD_VERDICT", repr(verdict)))
        verdicts[(row.get("id") or "").strip()] = (verdict, (row.get("comment") or "").strip())
    if bad:
        raise DataValidationError(bad)
    return verdicts


def _has_verdicts(path: Path) -> bool:
    try:
        return any(verdict for verdict, _ in read_verdicts(path).values())
    except (OSError, QFError):
        return path.exists()


# --- import of the verdicts ----------------------------------------------------------------


@dataclass(frozen=True)
class ReviewResult:
    kept: list[Candidate]  # ok (reviewed=True) and not yet reviewed (reviewed=False)
    fixes: list[tuple[Candidate, str]]
    drops: list[Candidate]

    @property
    def unreviewed(self) -> list[str]:
        return [c.record.id for c in self.kept if not c.record.reviewed]


def import_review(
    candidates: Sequence[Candidate], verdicts: Mapping[str, tuple[str, str]]
) -> ReviewResult:
    """ok -> reviewed=True; fix / drop -> removed (fix is listed for the generator); blank ->
    kept as unreviewed. Ids that are not candidates are an error; the agent never edits gold."""
    known = {c.record.id for c in candidates}
    unknown = sorted(set(verdicts) - known)
    if unknown:
        raise DataValidationError(
            [Issue(record_id, None, "UNKNOWN_ID", "not a candidate") for record_id in unknown]
        )
    kept, fixes, drops = [], [], []
    for candidate in candidates:
        verdict, comment = verdicts.get(candidate.record.id, ("", ""))
        if verdict == "ok":
            record = SFTRecord.model_validate({**candidate.record.model_dump(), "reviewed": True})
            kept.append(Candidate(candidate.slice, candidate.role, record))
        elif verdict == "fix":
            fixes.append((candidate, comment))
        elif verdict == "drop":
            drops.append(candidate)
        else:
            kept.append(candidate)
    return ReviewResult(kept=kept, fixes=fixes, drops=drops)


# --- stages --------------------------------------------------------------------------------


def _read(path: Path, root: Path, schema: str) -> tuple[ArtifactRef, list[SFTRecord]]:
    """Verified records that all answer in the benchmark's schema."""
    ref = read_artifact(path, "sft_dataset", supported_versions("sft_dataset"), root=root)
    records, issues = read_sft_records(root / ref.path)
    if issues:
        raise DataValidationError(issues)
    other = sorted({r.schema_version for r in records} - {schema})
    if other:
        raise QFError(f"{ref.path}: records in {', '.join(other)} do not belong in a {schema} "
                      "benchmark")  # fmt: skip
    return ref, records


def _candidates_of(path: Path, root: Path, schema: str) -> tuple[ArtifactRef, list[Candidate]]:
    ref, records = _read(path, root, schema)
    extra = load_artifact_manifest(root / ref.path, root=root).extra
    roles, slices = extra["roles"], extra["slices"]
    return ref, [Candidate(slices[r.id], roles[r.id], r) for r in records]


def _write_candidates(
    candidates: Sequence[Candidate], path: Path, *, run_id: str, parents: Sequence[str],
    root: Path, extra: Mapping[str, Any],
) -> ArtifactRef:  # fmt: skip
    return write_sft_records(
        [c.record for c in candidates], path, run_id=run_id, parents=parents, root=root,
        extra={**extra, "slices": {c.record.id: c.slice for c in candidates},
               "roles": {c.record.id: c.role for c in candidates}},
    )  # fmt: skip


@dataclass(frozen=True)
class ExportOutcome:
    candidates: list[Candidate]
    markdown_path: Path
    csv_path: Path
    run_dir: Path


def export_review(cfg: BenchmarkConfig, *, root: Path) -> ExportOutcome:
    """Select candidates from the verified test/test_ood files; write them with a readable
    Markdown and a CSV for verdicts. Refuses to overwrite a CSV that has verdicts."""
    if (root / cfg.bench_path).exists():
        raise QFError(f"{cfg.bench_path} is frozen; a new benchmark is a new version (bench_v2)")
    csv_path = root / cfg.review_csv
    if _has_verdicts(csv_path):
        raise QFError(f"{cfg.review_csv} already has verdicts; move it away to start over")
    run = start_run("bench-export", root)
    refs, datasets = {}, {}
    for split in ("test", "test_ood"):
        refs[split], datasets[split] = _read(root / cfg.source_dir / f"{split}.jsonl", root,
                                             cfg.schema_version)  # fmt: skip
    candidates = select_candidates(datasets, cfg)
    work = root / cfg.work_dir
    ref = _write_candidates(candidates, work / "candidates.jsonl", run_id=run.run_id,
                            parents=[r.sha256 for r in refs.values()], root=root,
                            extra={"version": cfg.version})  # fmt: skip
    markdown_path = work / f"review_{cfg.version}.md"
    atomic_write_text(markdown_path, render_review_markdown(candidates, cfg.version))
    atomic_write_text(csv_path, _review_csv(candidates))
    run.finish(config=cfg.model_dump(mode="json"), metrics={"candidates": len(candidates)},
               data_hashes={**{r.path.as_posix(): r.sha256 for r in refs.values()},
                            ref.path.as_posix(): ref.sha256})  # fmt: skip
    return ExportOutcome(candidates, markdown_path, csv_path, run.run_dir)


def import_review_stage(cfg: BenchmarkConfig, csv_path: Path, *, root: Path) -> ReviewResult:
    """Apply the verdicts of `csv_path`: `reviewed.jsonl` and `fix_list.md` in the work dir."""
    run = start_run("bench-review", root)
    source, candidates = _candidates_of(root / cfg.work_dir / "candidates.jsonl", root,
                                        cfg.schema_version)  # fmt: skip
    result = import_review(candidates, read_verdicts(csv_path))
    ref = _write_candidates(
        result.kept, root / cfg.work_dir / "reviewed.jsonl", run_id=run.run_id,
        parents=[source.sha256], root=root,
        extra={"version": cfg.version, "review_csv_sha256": sha256_file(csv_path)},
    )  # fmt: skip
    fix_lines = [f"# Записи {cfg.version} с вердиктом fix (исправить генератор)", ""]
    fix_lines += [f"- `{c.record.id}` ({c.slice}): {comment or 'без комментария'}"
                  for c, comment in result.fixes] or ["Нет."]  # fmt: skip
    atomic_write_text(root / cfg.work_dir / "fix_list.md", "\n".join(fix_lines) + "\n")
    run.finish(
        data_hashes={source.path.as_posix(): source.sha256, ref.path.as_posix(): ref.sha256},
        metrics={"ok": len(result.kept) - len(result.unreviewed), "fix": len(result.fixes),
                 "drop": len(result.drops), "unreviewed": len(result.unreviewed)},
    )  # fmt: skip
    return result


@dataclass(frozen=True)
class FreezeOutcome:
    bench: ArtifactRef
    counts: dict[str, int]  # slice -> records in the benchmark
    shortfalls: dict[str, int]  # slice -> records missing to its target
    manual: int
    split_refs: dict[str, ArtifactRef]
    warnings: list[str]
    run_dir: Path


def _choose(
    cfg: BenchmarkConfig, kept: Sequence[Candidate], allow_unreviewed: bool
) -> list[Candidate]:
    """Per slice, the first `count` usable records in priority order (main before reserve)."""
    chosen = []
    for spec in cfg.slices:
        usable = [
            c for c in kept if c.slice == spec.name and (c.record.reviewed or allow_unreviewed)
        ]
        chosen += usable[: spec.count]
    return chosen


def freeze(
    cfg: BenchmarkConfig, *, root: Path, manual: Sequence[SFTRecord] = (),
    manual_source: Path | None = None, allow_unreviewed: bool = False,
) -> FreezeOutcome:  # fmt: skip
    """Write `<version>.jsonl` (+ `.sha256`, a `benchmark` artifact) once, and the splits
    without the benchmark records into `out_dir`. Refuses unreviewed records unless allowed."""
    bench_path = root / cfg.bench_path
    if bench_path.exists():
        raise QFError(f"{cfg.bench_path} is already frozen; a new benchmark is bench_v2")
    source, kept = _candidates_of(root / cfg.work_dir / "reviewed.jsonl", root,
                                  cfg.schema_version)  # fmt: skip
    unreviewed = sum(not c.record.reviewed for c in kept if c.role == "main")
    if unreviewed and not allow_unreviewed:
        raise QFError(f"{unreviewed} main candidate(s) have no verdict yet; finish the review "
                      "or pass --allow-unreviewed (tests only)")  # fmt: skip
    chosen = _choose(cfg, kept, allow_unreviewed)
    warnings = []
    if allow_unreviewed and any(not c.record.reviewed for c in chosen):
        warnings.append(f"{sum(not c.record.reviewed for c in chosen)} unreviewed record(s) "
                        "frozen with --allow-unreviewed")  # fmt: skip
    records = [
        SFTRecord.model_validate({**c.record.model_dump(), "split": "bench"}) for c in chosen
    ]
    records += list(manual)
    foreign = [r.id for r in manual if r.schema_version != cfg.schema_version]
    if foreign:
        raise QFError(f"manual cases {foreign[:3]} are not {cfg.schema_version}")
    issues = [issue for record in records for issue in check_record(record)]
    if issues:
        raise DataValidationError(issues)
    return _write_frozen(cfg, root, source, chosen, records, len(manual), manual_source, warnings)


def _write_frozen(
    cfg: BenchmarkConfig, root: Path, source: ArtifactRef, chosen: Sequence[Candidate],
    records: Sequence[SFTRecord], manual: int, manual_source: Path | None, warnings: list[str],
) -> FreezeOutcome:  # fmt: skip
    run = start_run("bench-freeze", root)
    counts = {s.name: sum(c.slice == s.name for c in chosen) for s in cfg.slices}
    shortfalls = {s.name: s.count - counts[s.name] for s in cfg.slices if counts[s.name] < s.count}
    parents = [source.sha256] + ([sha256_file(manual_source)] if manual_source else [])
    bench = write_sft_records(
        records, root / cfg.bench_path, run_id=run.run_id, parents=parents, root=root,
        kind="benchmark", extra={"version": cfg.version, "slices": counts, "manual": manual,
                                 "shortfalls": shortfalls, "warnings": warnings},
    )  # fmt: skip
    atomic_write_text(root / cfg.out_dir / f"{cfg.version}.sha256",
                      f"{bench.sha256}  {cfg.bench_path.name}\n")  # fmt: skip
    frozen = {record.id for record in records}
    split_refs = {}
    for name in DATASET_FILES:
        path = root / cfg.source_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        ref, items = _read(path, root, cfg.schema_version)
        kept = [r for r in items if r.id not in frozen]
        extra = load_artifact_manifest(root / ref.path, root=root).extra
        # An unchanged copy has the same sha256 as its source: it takes the source's parents,
        # otherwise it would be its own parent and the lineage to load_facts would break.
        parents = list(ref.parents) if len(kept) == len(items) else [ref.sha256]
        split_refs[name] = write_sft_records(
            kept, root / cfg.out_dir / f"{name}.jsonl", run_id=run.run_id, parents=parents,
            root=root, extra={**extra, "count": len(kept), "without": cfg.version},
        )  # fmt: skip
    run.finish(
        data_hashes={bench.path.as_posix(): bench.sha256,
                     **{r.path.as_posix(): r.sha256 for r in split_refs.values()}},
        metrics={"records": len(records), "slices": counts, "manual": manual,
                 "shortfalls": shortfalls},
    )  # fmt: skip
    return FreezeOutcome(bench, counts, shortfalls, manual, split_refs, warnings, run.run_dir)


def verify_benchmark(path: Path, *, root: Path) -> ArtifactRef:
    """The file must match its `.sha256` file and its artifact manifest."""
    ref = read_artifact(path, "benchmark", supported_versions("benchmark"), root=root)
    sha_file = (root / ref.path).with_suffix(".sha256")
    if not sha_file.exists():
        raise QFError(f"{sha_file.name} is missing")
    expected = sha_file.read_text(encoding="utf-8").split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != ref.sha256:
        raise QFError(f"{ref.path}: sha256 differs from {sha_file.name}")
    return ref
