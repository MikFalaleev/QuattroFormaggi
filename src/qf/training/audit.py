"""The mask audit of step 11: what the model is trained on, shown decoded, plus the token
budget of the data (for the estimate of step 12)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qf.contracts import SFTRecord
from qf.training.features import (
    IGNORE_INDEX,
    ChatTokenizer,
    Features,
    Skip,
    TemplateCheck,
    build_features,
    check_template,
)

__all__ = ["AuditExample", "AuditReport", "audit_masks", "render_audit"]


@dataclass(frozen=True)
class AuditExample:
    record_id: str
    prompt: str  # the decoded prompt (everything with label -100)
    trainable: str  # the decoded trainable tokens: must be the answer JSON + EOS
    answer: str  # the gold answer text
    prompt_tokens: int
    trainable_tokens: int
    template: TemplateCheck

    @property
    def trainable_is_answer(self) -> bool:
        return self.trainable == self.answer + "</s>"


def _percentile(values: Sequence[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(share * (len(ordered) - 1)))]


@dataclass
class AuditReport:
    max_len: int
    examples: list[AuditExample]
    records: int = 0
    skipped: list[Skip] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    trainable_total: int = 0
    template: Counter[str] = field(default_factory=Counter)  # outcome of check_template
    # split -> [records, tokens, trainable tokens]: the token budget of each split (step 12)
    per_split: dict[str, list[int]] = field(default_factory=dict)

    @property
    def too_long_rate(self) -> float:
        return len(self.skipped) / self.records if self.records else 0.0

    def summary(self) -> dict[str, Any]:
        total = sum(self.lengths)
        return {
            "records": self.records, "max_len": self.max_len,
            "too_long": len(self.skipped), "too_long_rate": round(self.too_long_rate, 6),
            "tokens_p50": _percentile(self.lengths, 0.5) if self.lengths else None,
            "tokens_p95": _percentile(self.lengths, 0.95) if self.lengths else None,
            "tokens_max": max(self.lengths) if self.lengths else None,
            "tokens_total": total, "trainable_total": self.trainable_total,
            "trainable_share": round(self.trainable_total / total, 4) if total else None,
            "template": dict(self.template),
            "examples_trainable_is_answer": all(e.trainable_is_answer for e in self.examples),
            "per_split": {split: {"records": v[0], "tokens": v[1], "trainable": v[2]}
                          for split, v in self.per_split.items()},
        }  # fmt: skip


def _example(record: SFTRecord, features: Features, tok: ChatTokenizer) -> AuditExample:
    trainable = [t for t in features.labels if t != IGNORE_INDEX]
    return AuditExample(
        record_id=record.id,
        prompt=tok.decode(features.input_ids[: features.prompt_tokens]),
        trainable=tok.decode(trainable), answer=record.messages[2].content,
        prompt_tokens=features.prompt_tokens, trainable_tokens=len(trainable),
        template=check_template(record, tok),
    )  # fmt: skip


def audit_masks(records: Sequence[SFTRecord], tok: ChatTokenizer, *, max_len: int,
                n_examples: int = 5) -> AuditReport:  # fmt: skip
    """Features of every record (errors raise), the template check of every record, and the
    decoded masks of the first `n_examples` that fit."""
    report = AuditReport(max_len=max_len, examples=[])
    for record in records:
        report.records += 1
        built = build_features(record, tok, max_len)
        check = check_template(record, tok)
        report.template["identical" if check.identical else check.difference] += 1
        if not check.answer_tokens_match:
            report.template["answer tokens differ from the template"] += 1
        if isinstance(built, Skip):
            report.skipped.append(built)
            report.lengths.append(built.tokens)
            continue
        report.lengths.append(len(built.input_ids))
        report.trainable_total += built.answer_tokens
        split = report.per_split.setdefault(record.split or "—", [0, 0, 0])
        split[0] += 1
        split[1] += len(built.input_ids)
        split[2] += built.answer_tokens
        if len(report.examples) < n_examples:
            report.examples.append(_example(record, built, tok))
    return report


_DIFFERENCES_RU = {
    "identical": "совпадает",
    "the template drops the system prompt from the finished dialogue":
        "шаблон теряет системный промпт в готовом диалоге (вариант B его сохраняет)",
    "whitespace only": "отличаются только пробелы",
    "answer tokens differ from the template": "ТОКЕНЫ ОТВЕТА ОТЛИЧАЮТСЯ ОТ ШАБЛОНА",
}  # fmt: skip


def _ru(outcome: str) -> str:
    return _DIFFERENCES_RU.get(outcome, outcome)


def _fence(text: str) -> list[str]:
    fence = "`" * max(3, max((len(r) for r in text.split("\n") if set(r) == {"`"}), default=0) + 1)
    return [f"{fence}text", text, fence]


def render_audit(report: AuditReport, meta: dict[str, Any]) -> str:
    """`mask_audit.md`: the tokenizer, the statistics and the decoded examples."""
    s = report.summary()
    lines = [
        "# Аудит маски обучения (шаг 11)", "",
        f"- Модель: `{meta['model']}` @ `{meta['revision']}`",
        f"- tokenizer sha256 `{meta['tokenizer_hash']}`, шаблон чата sha256 "
        f"`{meta['chat_template_hash']}`",
        f"- Данные: {meta['data']} — {s['records']} записей; `max_len` {s['max_len']}",
        "- Способ построения: вариант B — промпт как при работе модели "
        "(`apply_chat_template(messages[:2], add_generation_prompt=True)`), затем токены ответа "
        "и один EOS; labels = −100 на промпте.", "",
        "## Итоги", "",
        f"- Длиннее `max_len` (пропущены, не обрезаны): {s['too_long']} "
        f"({s['too_long_rate']:.2%})",
        f"- Длина записи в токенах: p50 {s['tokens_p50']}, p95 {s['tokens_p95']}, "
        f"максимум {s['tokens_max']}",
        f"- Всего токенов {s['tokens_total']}, обучаемых {s['trainable_total']} "
        f"({s['trainable_share']:.1%})",
        "- Сверка с шаблоном полного диалога: "
        + "; ".join(f"{_ru(k)} — {v}" for k, v in s["template"].items()),
        "- По выборкам (записей / токенов / обучаемых): "
        + "; ".join(f"{k} {v['records']} / {v['tokens']} / {v['trainable']}"
                    for k, v in s["per_split"].items()),
        f"- В примерах обучаемые токены = JSON ответа + `</s>`: "
        f"{'да' if s['examples_trainable_is_answer'] else 'НЕТ'}", "",
    ]  # fmt: skip
    for n, e in enumerate(report.examples, start=1):
        template = _ru("identical" if e.template.identical else e.template.difference)
        answer_tokens = "те же" if e.template.answer_tokens_match else "ДРУГИЕ"
        matches = "да" if e.trainable_is_answer else "НЕТ"
        lines += [
            f"## Пример {n}: `{e.record_id}`", "",
            f"Промпт — {e.prompt_tokens} токенов (не обучаются); ответ — {e.trainable_tokens} "
            f"токенов (обучаются). Сверка с шаблоном: {template}; токены ответа в шаблоне "
            f"{answer_tokens}.", "",
            "Промпт (раскодирован):", "", *_fence(e.prompt), "",
            "Обучаемые токены (раскодированы):", "", *_fence(e.trainable), "",
            f"Совпадает с ответом + `</s>`: {matches}", "",
        ]  # fmt: skip
    return "\n".join(lines) + "\n"
