"""Inputs of the acceptance runs of the eval: fake answers to a frozen benchmark. NOT a model.

    uv run python -m tests.fake_eval        # bench_v1 (step 9) -> data/processed/fake_eval/
    uv run python -m tests.fake_eval --bench data/splits_v2/bench_v2.jsonl   # sub-step V6
    uv run qf eval run --config data/processed/fake_eval/fake_bench_v1.yaml
    uv run qf eval run --config data/processed/fake_eval/fake_bench_v1_gold.yaml
    uv run qf eval compare runs/<run of the first> runs/<run of the second>

`fake_<bench>` answers with the gold answers of the benchmark, spoiled in a fixed pattern
(`mistake`): timeouts, truncated JSON, `pieces: 1` invented where pieces are missing,
origin and destination swapped, code fences; for card_v2 also the special conditions
(`_condition_mistake`): values filled in where the gold asks for them, a temperature invented
for a reefer without one, a condition dropped, a temperature added to a tent or a van, a
wrong temperature, and conditions reordered with lists reversed and dimensions in other units
(which must still score as right). Every number of the report can be checked by hand against
the counts this module prints. `fake_<bench>_gold` answers with the gold answers unchanged.
The files stay outside `configs/eval`: they are not a model configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

import yaml

from qf.common import DATA_PROCESSED, DATA_SPLITS, project_root, sha256_text

DEFAULT_BENCH: Final = DATA_SPLITS / "bench_v1.jsonl"
DEFAULT_OUT: Final = DATA_PROCESSED / "fake_eval"
# The metrics of the step 9 run on bench_v1: this list stays as committed, so that run stays
# byte-reproducible (tests/test_v1_frozen.py).
METRICS: Final = [
    "json_valid_rate", "lenient_parse_rate", "failed_output_rate", "key_field_accuracy",
    "field_accuracy_micro", "field_accuracy.origin", "field_accuracy.destination",
    "field_accuracy.pickup_date", "field_accuracy.delivery_date", "field_accuracy.equipment_type",
    "field_accuracy.cargo_category", "field_accuracy.pieces", "field_accuracy.weight_total",
    "field_accuracy.weight_per_piece", "field_accuracy.temperature_c",
    "field_accuracy.shipper_name", "missing_f1", "missing_exact_rate", "conflict_recall",
    "hallucination_rate", "critical_error_count", "critical_error_count.pieces_weight_swap",
    "critical_error_count.wrong_unit_magnitude", "critical_error_count.origin_destination_swap",
    "critical_error_count.hallucinated_required_field", "critical_error_count.missed_conflict",
]  # fmt: skip
SLICE_METRICS: Final = [
    "json_valid_rate", "key_field_accuracy", "field_accuracy_micro", "missing_f1",
    "conflict_recall", "hallucination_rate", "critical_error_count",
]  # fmt: skip
_CONDITION_KINDS: Final = ("temperature", "securing", "packaging", "oversize", "sensors")
METRICS_V2: Final = [
    *(m for m in METRICS if m != "field_accuracy.temperature_c"),
    "field_accuracy.special_conditions", "special_conditions_exact_rate",
    *(f"field_accuracy.special_conditions.{kind}" for kind in _CONDITION_KINDS),
    "condition_precision", "condition_recall",
    *(f"condition_precision.{kind}" for kind in _CONDITION_KINDS),
    *(f"condition_recall.{kind}" for kind in _CONDITION_KINDS),
    "critical_error_count.hallucinated_condition", "critical_error_count.missed_condition",
]  # fmt: skip
SLICE_METRICS_V2: Final = [
    *SLICE_METRICS, "special_conditions_exact_rate", "condition_precision", "condition_recall",
]  # fmt: skip
METRICS_BY_SCHEMA: Final = {"card_v1": (METRICS, SLICE_METRICS),
                            "card_v2": (METRICS_V2, SLICE_METRICS_V2)}  # fmt: skip
_FILLED: Final[dict[str, dict[str, Any]]] = {
    "temperature": {"min_c": 2, "max_c": 6},
    "securing": {"methods": ["straps"]},
    "packaging": {"types": ["crate"]},
    "oversize": {"length": {"value": 9, "unit": "m"}, "width": {"value": 3, "unit": "m"},
                 "height": {"value": 4, "unit": "m"}},
    "sensors": {"parameters": ["temperature"]},
}  # fmt: skip


def _compact(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def mistake(index: int, gold_text: str) -> tuple[str, str]:
    """(kind, answer) for the record at `index` of the benchmark."""
    target = json.loads(gold_text)
    card = target["card"]
    if index % 47 == 7:
        return "timeout", "__timeout__"
    if index % 31 == 5:
        return "invalid_json", gold_text[: len(gold_text) // 2]
    if "pieces" in target["missing_fields"] and not target["conflicts"] and index % 2 == 0:
        card["pieces"] = 1
        target["missing_fields"].remove("pieces")
        return "invented_pieces", _compact(target)
    if index % 17 == 3 and card["origin"] and card["destination"]:
        card["origin"], card["destination"] = card["destination"], card["origin"]
        return "od_swap", _compact(target)
    if "special_conditions" in card and (kind := _condition_mistake(index, target)):
        return kind, _compact(target)
    if index % 10 == 1:
        return "code_fence", f"```json\n{gold_text}\n```"
    return "gold", gold_text


def _other_units(length: dict[str, Any] | None) -> dict[str, Any] | None:
    if length is None:
        return None
    if length["unit"] == "m":
        return {"value": round(length["value"] * 100, 6), "unit": "cm"}
    return {"value": round(length["value"] / 100, 6), "unit": "m"}


def _condition_mistake(index: int, target: dict[str, Any]) -> str | None:
    """A mistake in the special conditions of a card_v2 answer (changes `target`), or None."""
    card, missing = target["card"], target["missing_fields"]
    conditions: list[dict[str, Any]] = card["special_conditions"]
    asked = [c for c in conditions if f"special_conditions.{c['kind']}" in missing]
    if asked and index % 2 == 1:  # no values or a part of the dimensions -> fill them in
        condition = asked[0]
        for name, value in _FILLED[condition["kind"]].items():
            if not condition[name]:
                condition[name] = copy.deepcopy(value)
        missing.remove(f"special_conditions.{condition['kind']}")
        return "filled_condition_values"
    kinds = [c["kind"] for c in conditions]
    if "special_conditions.temperature" in missing and "temperature" not in kinds and index % 2:
        conditions.insert(0, {"kind": "temperature", **_FILLED["temperature"]})
        missing.remove("special_conditions.temperature")
        return "invented_required_temperature"
    if conditions and index % 9 == 2:
        dropped = conditions.pop(0)
        if f"special_conditions.{dropped['kind']}" in missing:
            missing.remove(f"special_conditions.{dropped['kind']}")
        return "dropped_condition"
    if not conditions and card["equipment_type"] in ("tent", "van") and index % 9 == 4:
        conditions.append({"kind": "temperature", **_FILLED["temperature"]})
        return "added_temperature"
    temperature = next((c for c in conditions if c["kind"] == "temperature"), None)
    if temperature and temperature["max_c"] is not None and index % 9 == 6:
        temperature["max_c"] += 1
        return "wrong_temperature"
    if index % 9 == 8 and conditions:  # the same answer in another order and other units
        conditions.reverse()
        for condition in conditions:
            for name, value in condition.items():
                if isinstance(value, list):
                    value.reverse()
                elif name in ("length", "width", "height"):
                    condition[name] = _other_units(value)
        return "reordered_same"
    return None


def _config(name: str, model: str, responses: Path, bench: Path, sha256: str,
            schema: str) -> dict[str, Any]:  # fmt: skip
    metrics, slice_metrics = METRICS_BY_SCHEMA[schema]
    return {
        "name": name,
        "backend": {"name": "fake", "model": model, "responses_file": responses.as_posix()},
        "bench": {"path": bench.as_posix(), "sha256": sha256},
        "generation": {"max_tokens": 768, "temperature": 0.0},
        "metrics": metrics,
        "slice_metrics": slice_metrics,
    }


def build(root: Path, bench: Path = DEFAULT_BENCH, out: Path = DEFAULT_OUT) -> Counter[str]:
    """Write the answers and the two eval configs under `root / out`; returns mistake counts."""
    sha256 = (root / bench).with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    stem = bench.stem
    (root / out).mkdir(parents=True, exist_ok=True)
    spoiled: dict[str, str] = {}
    gold: dict[str, str] = {}
    kinds: Counter[str] = Counter()
    lines = (root / bench).read_text(encoding="utf-8").splitlines()
    schema = json.loads(lines[0])["schema_version"]
    for index, line in enumerate(lines):
        messages = json.loads(line)["messages"]
        key = sha256_text(messages[1]["content"])
        kind, spoiled[key] = mistake(index, messages[2]["content"])
        gold[key] = messages[2]["content"]
        kinds[kind] += 1
    for suffix, model, answers in (("", "gold-with-mistakes", spoiled), ("_gold", "gold", gold)):
        responses = out / f"fake_responses_{stem}{suffix}.json"
        (root / responses).write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        config = _config(f"fake_{stem}{suffix}", model, responses, bench, sha256, schema)
        (root / out / f"fake_{stem}{suffix}.yaml").write_text(
            "# Fake answers (NOT a model): the step 9 acceptance run, see tests/fake_eval.py\n"
            + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return kinds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    kinds = build(project_root(), args.bench, args.out)
    print("Answers:", ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())))
    print(f"Configs: {args.out / f'fake_{args.bench.stem}.yaml'}, "
          f"{args.out / f'fake_{args.bench.stem}_gold.yaml'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
