"""Markdown reports of eval runs (plan step 9): one run (`render_report`) and a paired
comparison of two runs on the same benchmark (`compare_runs`).

The metric names come from the run (its config); this module holds no list of metrics.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Final

from pydantic import BaseModel

from qf.common import QFError
from qf.contracts import AnyFieldName, CaseScore, Place, Quantity, SFTRecord
from qf.domain import TargetParseError, TargetSchema, get_target_schema
from qf.eval.metrics import METRICS, Scorer, get_scorer
from qf.eval.results import EvalRun
from qf.eval.slices import record_kind, slice_of
from qf.eval.stats import mcnemar, paired_bootstrap_diff, resample_indices

__all__ = [
    "COMPARABLE_KEYS",
    "FAKE_BANNER",
    "MOCK_WARNING",
    "SYNTHETIC_WARNING",
    "compare_runs",
    "render_report",
]

FAKE_BANNER: Final = (
    "> **FAKE backend — это не результат модели.** Ответы взяты из подготовленного файла; "
    "прогон проверяет harness, метрики и отчёт, а не качество извлечения."
)
SYNTHETIC_WARNING: Final = (
    "benchmark полностью синтетический и шаблонный; качество на реальных заявках не измерено"
)
MOCK_WARNING: Final = (
    "benchmark синтетический: шаблонные заявки и заявки-моки, написанные агентом (D-103); "
    "качество на реальных заявках не измерено"
)
"""Instead of SYNTHETIC_WARNING when the synthetic benchmark holds mocks (not templates)."""
# Runs are comparable only on the same benchmark, prompt and answer schema (docs/EVAL_SPEC.md).
COMPARABLE_KEYS: Final = ("bench_sha256", "prompt_sha256", "schema_version")
TOP_FAILURES: Final = 20
OUTPUT_EXCERPT: Final = 600
CHANGED_LIST_LIMIT: Final = 50


def _number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}"
    return str(int(value)) if isinstance(value, float) else str(value)


def _interval(ci: Sequence[float] | None) -> str:
    return f"[{_number(ci[0])}, {_number(ci[1])}]" if ci else "—"


def _arrow(higher_is_better: bool) -> str:
    return "↑" if higher_is_better else "↓"


def _generation(settings: dict[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in settings.items())


def _header(run: EvalRun) -> list[str]:
    state = run.state
    failed = sum(s.failed for s in run.scores)
    return [
        f"- Прогон: `{run.run_id}`",
        f"- Backend: `{state['backend']}`, модель: `{state['model_id']}`",
        f"- Benchmark: `{state['bench_path']}`, sha256 `{state['bench_sha256']}`",
        f"- Схема ответа: `{state['schema_version']}`, системный промпт sha256 "
        f"`{state['prompt_sha256']}`",
        f"- Генерация: {_generation(state['generation'])}",
        f"- Кейсов: {len(run.scores)}, неуспешных ответов (ошибка генерации или не разобран "
        f"даже мягко): {failed}",
    ]  # fmt: skip


def _overall(table: dict[str, Any]) -> list[str]:
    bootstrap = table["bootstrap"]
    lines = [
        "## Метрики", "",
        f"95% доверительные интервалы — bootstrap ({bootstrap['n']} выборок, seed "
        f"{bootstrap['seed']}); n — число кейсов, к которым метрика применима. "
        "↑ — чем больше, тем лучше; ↓ — наоборот.", "",
        "| Метрика | Значение | 95% CI | n | |", "|---|---:|---|---:|---|",
    ]  # fmt: skip
    for name, entry in table["overall"].items():
        lines.append(f"| `{name}` | {_number(entry['value'])} | {_interval(entry.get('ci'))} "
                     f"| {entry['n']} | {_arrow(entry['higher_is_better'])} |")  # fmt: skip
    return lines


def _error_counter(error: str) -> Callable[[CaseScore], int]:
    return lambda s: 0 if s.failed else s.critical_errors.count(error)


def _critical_by_kind(records: Sequence[SFTRecord], scores: Sequence[CaseScore],
                      scorer: Scorer) -> list[str]:  # fmt: skip
    """Counts of each critical error per kind of case (the columns of the schema's scorer);
    failed answers are not assessable. The hard slice of the release gates: card_v1 — missing +
    hard (D-079), card_v2 — everything but clean (D-085)."""
    kinds = {r.id: record_kind(r) for r in records}
    columns = dict(scorer.kind_columns)
    lines = [
        "## Критические ошибки по видам кейсов", "", scorer.kinds_note, "",
        "| Ошибка | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns),
    ]  # fmt: skip

    def row(label: str, count: Callable[[CaseScore], int]) -> str:
        cells = [sum(count(s) for s in scores if groups is None or kinds[s.record_id] in groups)
                 for groups in columns.values()]  # fmt: skip
        return f"| {label} | " + " | ".join(map(str, cells)) + " |"

    for error in scorer.critical_errors:
        lines.append(row(f"`{error}`", _error_counter(error)))
    lines.append(row("не оценено", lambda s: int(s.failed)))
    return lines


def _slices(table: dict[str, Any], metric_names: Sequence[str]) -> list[str]:
    lines = ["## Срезы", "", "Значения без интервалов; «—» — метрика к срезу не применима."]
    for slice_name, groups in table["slices"].items():
        lines += ["", f"### {slice_name}", "",
                  "| Группа | Кейсов | " + " | ".join(f"`{m}`" for m in metric_names) + " |",
                  "|---|---:|" + "---:|" * len(metric_names)]  # fmt: skip
        for group, entry in groups.items():
            cells = " | ".join(_number(entry[m]["value"]) for m in metric_names)
            lines.append(f"| {group} | {entry['cases']} | {cells} |")
    return lines


def _readable(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Quantity):
        return f"{_number(float(value.value))} {value.unit}"
    if isinstance(value, Place):
        return f"{value.city} ({value.region})"
    if isinstance(value, list):  # special conditions of card_v2
        return "[" + "; ".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                               if isinstance(item, BaseModel) else str(item)
                               for item in value) + "]"  # fmt: skip
    return str(value)


def _answer(raw: str, schema: TargetSchema[Any]) -> Any:
    try:
        return schema.parse_lenient(raw)
    except TargetParseError:
        return None


def _format_problem(raw: str, schema: TargetSchema[Any]) -> str | None:
    try:
        schema.parse(raw)
    except TargetParseError as exc:
        return f"строгий разбор не прошёл ({exc.stage}): {exc}"
    return None


def _problems(score: CaseScore, gold: Any, raw: str, schema: TargetSchema[Any],
              fields: Sequence[AnyFieldName]) -> list[str]:  # fmt: skip
    problems = []
    if score.generation_error is not None:
        problems.append(f"ошибка генерации: {score.generation_error}")
    elif (format_problem := _format_problem(raw, schema)) is not None:
        problems.append(format_problem)
    if score.critical_errors:
        problems.append("критические ошибки: " + ", ".join(score.critical_errors))
    answer = _answer(raw, schema)
    if answer is not None:
        wrong = [f for f in fields if score.field_correct.get(f) is False]
        for field in wrong:
            gold_value, answer_value = getattr(gold.card, field), getattr(answer.card, field)
            problems.append(f"`{field}`: эталон {_readable(gold_value)} → "
                            f"ответ {_readable(answer_value)}")  # fmt: skip
        if not score.missing_exact:
            problems.append(f"missing_fields: эталон {gold.missing_fields} → "
                            f"ответ {answer.missing_fields}")  # fmt: skip
        if score.conflict_detected is False:
            problems.append(f"конфликт не отмечен: эталон {[c.field for c in gold.conflicts]}")
    if score.notes:
        problems.append("заметки: " + ", ".join(score.notes))
    return problems


def _code_block(text: str) -> list[str]:
    """A fenced block that `text` cannot close: the fence is longer than any run of backticks
    inside (a model answer is often itself wrapped in ```json)."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", text, fence]


def _severity(score: CaseScore) -> tuple[int, int, int, int]:
    wrong = sum(v is False for v in score.field_correct.values())
    return (int(score.failed), len(score.critical_errors), wrong,
            int(not score.missing_exact) + int(not score.schema_valid))  # fmt: skip


def _failures(run: EvalRun, scorer: Scorer) -> list[str]:
    records = {r.id: r for r in run.records}
    schema = get_target_schema(scorer.version)
    ranked = sorted((s for s in run.scores if any(_severity(s))), key=_severity, reverse=True)
    lines = [f"## Худшие кейсы (топ-{TOP_FAILURES} из {len(ranked)} с ошибками)", "",
             "Порядок: неуспешные ответы, затем по числу критических ошибок, неверных полей, "
             "ошибок missing_fields и формата."]  # fmt: skip
    for n, score in enumerate(ranked[:TOP_FAILURES], start=1):
        record = records[score.record_id]
        raw = run.predictions[score.record_id]["output"]
        gold = schema.parse(record.messages[2].content)
        hard = slice_of(record, "hard_case")
        excerpt = raw if len(raw) <= OUTPUT_EXCERPT else raw[:OUTPUT_EXCERPT] + " …"
        lines += ["", f"### {n}. `{score.record_id}` — {hard}, {record.template_family}, "
                      f"{record.language}", "",
                  *(f"- {p}" for p in _problems(score, gold, raw, schema, scorer.fields)), "",
                  "Заявка:", "", *_code_block(record.messages[1].content), "",
                  "Ответ:", "", *_code_block(excerpt or "(пусто)")]  # fmt: skip
    return lines


def render_report(run: EvalRun) -> str:
    """The report of one run: header, metrics with CI, critical errors, slices, worst cases."""
    scorer = get_scorer(run.state["schema_version"])
    lines = [f"# Оценка: {run.state['name']}", ""]
    if run.state["backend"] == "fake":
        lines += [FAKE_BANNER, ""]
    if run.records and all(r.synthetic for r in run.records):
        mocks = any(r.template_family == "manual_mock" for r in run.records)
        lines += [f"> {MOCK_WARNING if mocks else SYNTHETIC_WARNING}.", ""]
    lines += [*_header(run), "", *_overall(run.table), "",
              *_critical_by_kind(run.records, run.scores, scorer), "",
              *_slices(run.table, run.state["slice_metrics"]), "",
              *_failures(run, scorer)]  # fmt: skip
    return "\n".join(lines) + "\n"


# --- comparison ----------------------------------------------------------------------------


def _paired(a: EvalRun, b: EvalRun) -> list[tuple[CaseScore, CaseScore]]:
    for key in COMPARABLE_KEYS:
        if a.state.get(key) != b.state.get(key):
            raise QFError(f"runs are not comparable: {key} differs "
                          f"({a.state.get(key)} vs {b.state.get(key)})")  # fmt: skip
    by_id = {s.record_id: s for s in b.scores}
    ids = [s.record_id for s in a.scores]
    if len(set(ids)) != len(ids) or set(ids) != set(by_id) or len(by_id) != len(b.scores):
        raise QFError("runs are not comparable: they scored different cases")
    return [(s, by_id[s.record_id]) for s in a.scores]


def _verdict(diff: tuple[float, float, float] | None, higher_is_better: bool) -> str:
    if diff is None:
        return "—"
    _, low, high = diff
    if low > 0 or high < 0:
        return "B лучше" if (low > 0) == higher_is_better else "B хуже"
    return "не различимо"


def _comparison_header(a: EvalRun, b: EvalRun) -> list[str]:
    lines = ["# Сравнение прогонов", ""]
    if "fake" in (a.state["backend"], b.state["backend"]):
        lines += [FAKE_BANNER, ""]
    if a.state["backend"] != b.state["backend"]:
        lines += ["> **Разные backend.** Разница включает и среду выполнения (квантование, "
                  "сборку, декодирование); модели сравнивают внутри одного backend "
                  "(docs/EVAL_SPEC.md).", ""]  # fmt: skip
    if a.state["generation"] != b.state["generation"]:
        lines += ["> **Разные настройки генерации** — разница включает и их.", ""]
    lines += [
        "| | A | B |", "|---|---|---|",
        f"| Прогон | `{a.run_id}` | `{b.run_id}` |",
        f"| Backend | `{a.state['backend']}` | `{b.state['backend']}` |",
        f"| Модель | `{a.state['model_id']}` | `{b.state['model_id']}` |",
        f"| Генерация | {_generation(a.state['generation'])} | "
        f"{_generation(b.state['generation'])} |", "",
        f"Benchmark sha256 `{a.state['bench_sha256']}`, промпт sha256 "
        f"`{a.state['prompt_sha256']}`, схема `{a.state['schema_version']}` — у обоих прогонов "
        "одинаковые.",
    ]  # fmt: skip
    return lines


def compare_runs(a: EvalRun, b: EvalRun) -> str:
    """Paired differences B − A with bootstrap intervals, McNemar's test on key_fields_correct
    and the cases that were fixed or broken. Refuses runs on a different benchmark, prompt or
    schema; different backends are allowed and flagged."""
    pairs = _paired(a, b)
    bootstrap = a.table.get("bootstrap", {"n": 2000, "seed": 0})
    indices = resample_indices(len(pairs), bootstrap["n"], bootstrap["seed"])
    names = [m for m in a.state["metrics"] if m in b.state["metrics"]]
    lines = [
        *_comparison_header(a, b),
        "",
        "## Метрики",
        "",
        f"B − A и 95% CI разницы — парный bootstrap ({bootstrap['n']} выборок, seed "
        f"{bootstrap['seed']}) по одним и тем же кейсам.",
        "",
        "| Метрика | A | B | B − A | 95% CI | Вывод |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in names:
        metric = METRICS.get(name)()
        values_a = [metric.value(sa) for sa, _ in pairs]
        values_b = [metric.value(sb) for _, sb in pairs]
        diff = paired_bootstrap_diff(values_a, values_b, statistic=metric.aggregate,
                                     indices=indices)  # fmt: skip
        lines.append(
            f"| `{name}` | {_number(metric.aggregate(values_a))} | "
            f"{_number(metric.aggregate(values_b))} | {_number(diff[0] if diff else None)} | "
            f"{_interval(diff[1:] if diff else None)} | {_verdict(diff, metric.higher_is_better)} |"
        )  # fmt: skip
    key_pairs = [
        (sa, sb)
        for sa, sb in pairs
        if sa.key_fields_correct is not None and sb.key_fields_correct is not None
    ]
    flags_a = [bool(sa.key_fields_correct) for sa, _ in key_pairs]
    flags_b = [bool(sb.key_fields_correct) for _, sb in key_pairs]
    b01, b10, p_value = mcnemar(flags_a, flags_b)
    fixed = [
        sa.record_id for sa, sb in key_pairs if not sa.key_fields_correct and sb.key_fields_correct
    ]
    broken = [
        sa.record_id for sa, sb in key_pairs if sa.key_fields_correct and not sb.key_fields_correct
    ]
    lines += ["", "## McNemar по key_fields_correct", "",
              f"Кейсов с ключевыми полями: {len(key_pairs)}. Исправлено (A неверно, B верно): "
              f"{b01}; сломано (A верно, B неверно): {b10}; точный двусторонний p = "
              f"{p_value:.4g}.", ""]  # fmt: skip
    for title, ids in (("Исправлено", fixed), ("Сломано", broken)):
        shown = ", ".join(f"`{i}`" for i in ids[:CHANGED_LIST_LIMIT]) or "—"
        more = f" … и ещё {len(ids) - CHANGED_LIST_LIMIT}" if len(ids) > CHANGED_LIST_LIMIT else ""
        lines.append(f"- **{title}** ({len(ids)}): {shown}{more}")
    return "\n".join(lines) + "\n"
