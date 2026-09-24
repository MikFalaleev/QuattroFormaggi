"""Samples of a generated dataset for a person to read (sub-step V3 stop).

`curate` picks records greedily so that together they show every family, language, equipment
type, condition kind, hard case and length unit that occur in the data; `render_review` prints
each request with its gold answer in plain Russian and the count of every hard case per split.
The file is for reading only: it is not an artifact and nothing reads it back.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel

from qf.common import sha256_text
from qf.contracts import SFTRecord
from qf.data.conditions import LABELS_RU, describe_conditions
from qf.domain import get_target_schema

__all__ = ["curate", "render_review"]

_FIELDS_RU: Final = (
    ("shipper_name", "Отправитель"), ("cargo_category", "Категория"),
    ("equipment_type", "Транспорт"), ("pieces", "Мест"), ("weight_total", "Общий вес"),
    ("weight_per_piece", "Вес места"), ("origin", "Откуда"), ("destination", "Куда"),
    ("pickup_date", "Загрузка"), ("delivery_date", "Доставка"),
)  # fmt: skip


def _features(record: SFTRecord) -> set[str]:
    target = json.loads(record.messages[2].content)
    card = target["card"]
    conditions = card.get("special_conditions") or []
    features = {f"семейство {record.template_family}", f"язык {record.language}",
                f"трудный случай {','.join(record.variant.hard_cases) or 'нет'}"}  # fmt: skip
    if card.get("equipment_type"):
        features.add(f"транспорт {card['equipment_type']}")
    if card.get("cargo_category") == "machinery":
        features.add("спецтехника")
    features |= {f"условие {c['kind']}" for c in conditions}
    features |= {f"габариты в {d['unit']}" for c in conditions if c["kind"] == "oversize"
                 for d in (c["length"], c["width"], c["height"]) if d}  # fmt: skip
    features.add("есть условия" if conditions else "без условий")
    return features


def curate(records: Sequence[SFTRecord], *, seed: int, limit: int = 40) -> list[SFTRecord]:
    """Up to `limit` records chosen greedily to cover every feature of the data; once all are
    covered the next pass starts afresh, so each feature is shown several times. The order is
    seeded and does not depend on the order of `records`."""
    pool = sorted(records, key=lambda r: sha256_text(f"{seed}:{r.id}"))
    features = {r.id: _features(r) for r in pool}
    covered: set[str] = set()
    chosen: list[SFTRecord] = []
    while len(chosen) < limit and pool:
        best = max(pool, key=lambda r: len(features[r.id] - covered))
        if not features[best.id] - covered:
            if not covered:
                break
            covered = set()  # every feature shown: begin the next pass
            continue
        chosen.append(best)
        covered |= features[best.id]
        pool.remove(best)
    return chosen


def _readable(name: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict) and "unit" in value:
        return f"{value['value']} {value['unit']}"
    if isinstance(value, dict) and "city" in value:
        return f"{value['city']} ({value['region']})"
    return (
        LABELS_RU.get(str(value), str(value))
        if name in ("cargo_category", "equipment_type")
        else str(value)
    )


def _json(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _gold(record: SFTRecord) -> list[str]:
    schema = get_target_schema(record.schema_version)
    target = schema.parse(record.messages[2].content)
    card = target.card
    lines = [
        f"- {label}: {_readable(name, _json(getattr(card, name)))}" for name, label in _FIELDS_RU
    ]
    conditions = getattr(card, "special_conditions", None)
    if conditions is not None:
        yes = "да" if conditions else "нет"
        lines.append(f"- **Особые условия: {yes}**" + (f" — {describe_conditions(conditions)}"
                                                       if conditions else ""))  # fmt: skip
    lines.append(f"- **Недостающие:** {', '.join(target.missing_fields) or '—'}")
    conflicts = "; ".join(f"{c.field}: {c.values}" for c in target.conflicts)
    lines.append(f"- **Конфликты:** {conflicts or '—'}")
    return lines


def render_review(splits: Mapping[Any, Sequence[SFTRecord]], chosen: Sequence[SFTRecord],
                  dataset: str) -> str:  # fmt: skip
    counts = {name: Counter(",".join(r.variant.hard_cases) or "clean" for r in items)
              for name, items in splits.items()}  # fmt: skip
    kinds = sorted({kind for counter in counts.values() for kind in counter})
    lines = [f"# Примеры сгенерированных заявок ({dataset})", "",
             f"{len(chosen)} заявок, отобранных так, чтобы вместе показать каждое семейство, "
             "язык, тип транспорта, вид условия, трудный случай и единицу габаритов. Под "
             "каждой заявкой — правильный ответ, который построил код.", "",
             "## Трудные случаи по выборкам", "",
             "| Случай | " + " | ".join(splits) + " |",
             "|---|" + "---:|" * len(splits)]  # fmt: skip
    for kind in kinds:
        lines.append(f"| `{kind}` | " + " | ".join(str(counts[name][kind]) for name in splits)
                     + " |")  # fmt: skip
    for n, record in enumerate(chosen, start=1):
        hard = ", ".join(record.variant.hard_cases) or "без трудного случая"
        lines += ["", f"## {n}. {record.template_family}, {record.language} — {hard}", "",
                  f"`{record.id}`", "", "```text", record.messages[1].content, "```", "",
                  *_gold(record)]  # fmt: skip
    return "\n".join(lines) + "\n"
