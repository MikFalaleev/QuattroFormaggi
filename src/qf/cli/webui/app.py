"""What the local web interface can do (`qf ui`, D-107): read-only access to the data, the
benchmarks and the eval runs, scoring an answer typed by hand, and rendering a new request.

Every method returns plain JSON-ready values; the HTTP layer (`server.py`) only routes. The
interface never writes to the project: generated requests live in memory only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Final

from qf.common import RUNS, QFError, load_yaml_config
from qf.contracts import LoadFactsV2, OodReason, SFTRecord
from qf.data import (
    HARD_CASES,
    TEMPLATE_FAMILIES,
    GenerateConfig,
    read_sft_records,
    render_record,
)
from qf.domain import TargetParseError, get_target_schema, has_special_conditions
from qf.eval import get_scorer, record_kind, slice_of

__all__ = ["DATASETS", "GENERATED", "WebApp", "WebError"]


class WebError(QFError):
    """A request the interface cannot serve (unknown dataset, bad answer text...): HTTP 400."""


@dataclass(frozen=True)
class Dataset:
    id: str
    label: str
    path: str  # relative to the project root


def _splits(prefix: str, directory: str, label: str) -> list[Dataset]:
    return [Dataset(f"{prefix}/{split}", f"{label}: {split}", f"{directory}/{split}.jsonl")
            for split in ("test", "test_ood", "val", "train", "smoke")]  # fmt: skip


DATASETS: Final = (
    Dataset("bench_v2", "bench_v2 — замороженный benchmark card_v2",
            "data/splits_v2/bench_v2.jsonl"),
    *_splits("generated_v2", "data/processed/generated_v2", "generated_v2 (card_v2)"),
    Dataset("bench_v1", "bench_v1 — замороженный benchmark card_v1", "data/splits/bench_v1.jsonl"),
    *_splits("generated_v1", "data/processed/generated_v1", "generated_v1 (card_v1)"),
)  # fmt: skip
GENERATED: Final = "generated_here"
"""The dataset of the requests rendered in this session (kept in memory)."""
GENERATE_CONFIG: Final = "configs/data/generate_v2.yaml"
FACTS_V2: Final = "data/processed/load_facts_v2.jsonl"
RANDOM_TRIES: Final = 200
"""Random loads tried for a hard case that applies only to some loads (a lowbed, a reefer)."""
_RUN_ID: Final = re.compile(r"^\d{8}-\d{6}-[a-z0-9-]+$")
_LOAD_ID: Final = re.compile(r'"load_id":"([^"]+)"')


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _request_text(record: SFTRecord) -> str:
    return record.messages[1].content


@dataclass
class WebApp:
    root: Path
    _records: dict[str, list[SFTRecord]] = field(default_factory=dict)
    _facts_lines: dict[str, str] | None = None
    _config: GenerateConfig | None = None
    _sorted_loads: list[str] | None = None
    _counter: int = 0

    # --- data and benchmarks ------------------------------------------------------------------

    def datasets(self) -> list[dict[str, Any]]:
        """The datasets present on disk (and the requests rendered here, if any)."""
        found = [{"id": d.id, "label": d.label, "path": d.path}
                 for d in DATASETS if (self.root / d.path).exists()]  # fmt: skip
        if self._records.get(GENERATED):
            found.append({"id": GENERATED, "label": "созданные здесь заявки (в памяти)",
                          "path": ""})  # fmt: skip
        return found

    def _dataset(self, dataset_id: str) -> list[SFTRecord]:
        if dataset_id in self._records:
            return self._records[dataset_id]
        spec = next((d for d in DATASETS if d.id == dataset_id), None)
        if spec is None:
            raise WebError(f"неизвестный набор {dataset_id!r}")
        path = self.root / spec.path
        if not path.exists():
            raise WebError(f"нет файла {spec.path}: соберите данные (README, раздел «Команды»)")
        records, issues = read_sft_records(path)
        if issues:
            raise WebError(f"{spec.path}: {len(issues)} ошибок в записях, например {issues[0]}")
        self._records[dataset_id] = records
        return records

    def _record(self, dataset_id: str, record_id: str) -> SFTRecord:
        for record in self._dataset(dataset_id):
            if record.id == record_id:
                return record
        raise WebError(f"в наборе {dataset_id} нет записи {record_id!r}")

    def records(self, dataset_id: str) -> list[dict[str, Any]]:
        """Short rows for the list: what the filters of the page need."""
        return [self._summary(record) for record in self._dataset(dataset_id)]

    def _summary(self, record: SFTRecord) -> dict[str, Any]:
        card = get_target_schema(record.schema_version).parse(record.messages[2].content).card
        conditions = [c.kind for c in getattr(card, "special_conditions", ())]
        body = _request_text(record).split("\n\n", 1)[-1]
        return {
            "id": record.id, "schema": record.schema_version, "language": record.language,
            "family": record.template_family, "kind": record_kind(record),
            "hard_case": slice_of(record, "hard_case"), "equipment": card.equipment_type,
            "conditions": conditions, "synthetic": record.synthetic,
            "excerpt": " ".join(body.split())[:140],
        }  # fmt: skip

    def record(self, dataset_id: str, record_id: str) -> dict[str, Any]:
        return self._detail(self._record(dataset_id, record_id), dataset_id)

    def _detail(self, record: SFTRecord, dataset_id: str) -> dict[str, Any]:
        schema = get_target_schema(record.schema_version)
        gold = schema.parse(record.messages[2].content)
        return {
            **self._summary(record), "dataset": dataset_id, "text": _request_text(record),
            "variant": _dump(record.variant), "gold": _dump(gold),
            "gold_text": record.messages[2].content,
            "special_conditions_flag": _flag(gold.card),
            "source_id": record.source_id, "split": record.split,
        }  # fmt: skip

    # --- checking an answer -------------------------------------------------------------------

    def check(self, schema_version: str, answer: str) -> dict[str, Any]:
        """The answer parsed as a model answer would be: strict format, lenient parse, the rules
        of the schema (missing fields, order, conflicts) and its canonical form."""
        schema = _schema(schema_version)
        result: dict[str, Any] = {"schema": schema_version}
        try:
            schema.parse(answer)
            result["strict"] = {"ok": True, "error": None}
        except TargetParseError as exc:
            result["strict"] = {"ok": False, "error": f"{exc.stage}: {exc}"}
        try:
            target = schema.parse_lenient(answer)
        except TargetParseError as exc:
            result["lenient"] = {"ok": False, "error": f"{exc.stage}: {exc}"}
            return result
        result["lenient"] = {"ok": True, "error": None}
        card = schema.canonicalize(target.card)
        canonical = target.model_copy(update={"card": card})
        result.update(
            card=_dump(target.card), missing_fields=list(target.missing_fields),
            missing_by_rule=list(schema.missing_fields(target.card)),
            conflicts=[_dump(c) for c in target.conflicts],
            consistency=schema.check_consistency(target),
            canonical=schema.serialize(canonical),
            special_conditions_flag=_flag(target.card),
        )  # fmt: skip
        return result

    def score(self, dataset_id: str, record_id: str, answer: str) -> dict[str, Any]:
        """The answer scored against the gold of the record, as the eval harness would."""
        record = self._record(dataset_id, record_id)
        schema = get_target_schema(record.schema_version)
        scorer = get_scorer(record.schema_version)
        gold = schema.parse(record.messages[2].content)
        score = scorer.score(record.id, gold, answer)
        check = self.check(record.schema_version, answer)
        answer_card = check.get("card") or {}
        gold_card = _dump(gold.card)
        fields = [{"field": name, "gold": gold_card.get(name), "answer": answer_card.get(name),
                   "correct": score.field_correct.get(name)} for name in scorer.fields]  # fmt: skip
        return {
            "check": check, "score": _dump(score), "failed": score.failed, "fields": fields,
            "gold": _dump(gold), "kind": record_kind(record),
        }  # fmt: skip

    # --- rendering new requests ---------------------------------------------------------------

    def _facts(self) -> dict[str, str]:
        """load_id -> the JSON line of its facts (parsed only when a load is used)."""
        if self._facts_lines is None:
            path = self.root / FACTS_V2
            if not path.exists():
                raise WebError(f"нет {FACTS_V2}: запустите "
                               f"qf data build --config {GENERATE_CONFIG}")  # fmt: skip
            lines: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                match = _LOAD_ID.search(line)
                if match:
                    lines[match.group(1)] = line
            self._facts_lines = lines
        return self._facts_lines

    def _generate_config(self) -> GenerateConfig:
        if self._config is None:
            try:
                self._config = load_yaml_config(self.root / GENERATE_CONFIG, GenerateConfig)
            except QFError as exc:
                raise WebError(f"не читается конфиг генератора {GENERATE_CONFIG}: {exc}") from exc
        return self._config

    def generator_options(self) -> dict[str, Any]:
        loads = len(self._facts())
        cfg = self._generate_config()
        families = [{"name": name, "language": TEMPLATE_FAMILIES.get(name)().language,
                     "ood": name in cfg.ood.holdout_families}
                    for name in TEMPLATE_FAMILIES.names()]  # fmt: skip
        return {"families": families, "hard_cases": HARD_CASES.names(),
                "mix": cfg.mix, "loads": loads}  # fmt: skip

    def generate(self, load_id: str | None, family: str | None, kind: str | None,
                 seed: int) -> dict[str, Any]:  # fmt: skip
        """One request for a load rendered by the card_v2 generator: family and kind are drawn
        from the config where not given; `seed` gives another rendering of the same choice.
        Without `load_id` a random load is taken, one that the asked hard case applies to (up
        to `RANDOM_TRIES` loads); a load given by hand that the case does not apply to gets the
        request without it and a warning."""
        facts_lines = self._facts()
        cfg = self._generate_config()
        if load_id and load_id not in facts_lines:
            raise WebError(f"нет загрузки {load_id!r} (пример: LOAD00000001)")
        if family and family not in TEMPLATE_FAMILIES:
            raise WebError(f"нет семейства шаблонов {family!r}")
        allowed = {"clean", "hard", "conditions", *HARD_CASES.names()}
        if kind and kind not in allowed:
            raise WebError(f"нет вида заявки {kind!r}")
        ood: OodReason | None = "family" if family in cfg.ood.holdout_families else None
        rng = Random(seed)
        tries = 1 if load_id else RANDOM_TRIES
        for _ in range(tries):
            chosen = load_id or rng.choice(self._load_ids())
            facts = LoadFactsV2.model_validate_json(facts_lines[chosen])
            try:
                generated = render_record(facts, "test_ood" if ood else "test", seed, cfg,
                                          ood_reason=ood, source_prefix="ui",
                                          family=family or None, kind=kind or None)  # fmt: skip
            except QFError as exc:
                raise WebError(f"генератор не смог построить заявку: {exc}") from exc
            if _has_kind(generated.record.variant.hard_cases, kind, cfg):
                break
        warning = None
        if not _has_kind(generated.record.variant.hard_cases, kind, cfg):
            warning = (f"вид «{kind}» неприменим к загрузке {chosen}: заявка построена без "
                       "него (например, «не все габариты» бывают только у трала)")  # fmt: skip
        self._counter += 1
        record = generated.record.model_copy(update={"id": f"ui-{self._counter}-{chosen}"})
        self._records.setdefault(GENERATED, []).append(record)
        return {**self._detail(record, GENERATED), "facts": _dump(facts),
                "draft": _dump(generated.draft), "seed": seed, "warning": warning}  # fmt: skip

    def _load_ids(self) -> list[str]:
        if self._sorted_loads is None:
            self._sorted_loads = sorted(self._facts())
        return self._sorted_loads

    # --- eval runs ----------------------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        """Eval runs and comparisons under runs/, newest first."""
        found = []
        for run_dir in sorted(self._run_dirs(), reverse=True):
            state_path = run_dir / "eval_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                found.append({"id": run_dir.name, "type": "eval", "name": state.get("name"),
                              "backend": state.get("backend"), "model": state.get("model_id"),
                              "bench": state.get("bench_path"),
                              "schema": state.get("schema_version"),
                              "has_report": (run_dir / "report.md").exists()})  # fmt: skip
            elif (run_dir / "compare.md").exists():
                found.append({"id": run_dir.name, "type": "compare", "name": "сравнение",
                              "has_report": True})  # fmt: skip
        return found

    def _run_dirs(self) -> Iterable[Path]:
        runs = self.root / RUNS
        return (p for p in runs.iterdir() if p.is_dir()) if runs.exists() else ()

    def run_report(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID.match(run_id):
            raise WebError(f"неверный id прогона {run_id!r}")
        run_dir = self.root / RUNS / run_id
        for name in ("report.md", "compare.md"):
            if (run_dir / name).exists():
                return {"id": run_id, "file": name,
                        "markdown": (run_dir / name).read_text(encoding="utf-8")}  # fmt: skip
        raise WebError(f"у прогона {run_id} нет отчёта")


def _schema(version: str) -> Any:
    try:
        return get_target_schema(version)
    except QFError as exc:
        raise WebError(str(exc)) from exc


def _has_kind(hard_cases: list[str], kind: str | None, cfg: GenerateConfig) -> bool:
    """The record has what was asked: any kind for "" and "clean", a case of the group for
    "hard" and "conditions", the case itself otherwise."""
    if not kind or kind == "clean":
        return True
    if kind == "hard":
        return any(case in cfg.hard for case in hard_cases)
    if kind == "conditions":
        return any(case in cfg.condition_hard for case in hard_cases)
    return kind in hard_cases


def _flag(card: Any) -> str | None:
    """«да» / «нет» for card_v2 (computed by code, D-081); None for card_v1."""
    if not hasattr(card, "special_conditions"):
        return None
    return "да" if has_special_conditions(card) else "нет"
