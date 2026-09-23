"""Inputs of the step 9 acceptance run: fake answers to a frozen benchmark. NOT a model.

    uv run python -m tests.fake_eval        # writes data/processed/fake_eval/
    uv run qf eval run --config data/processed/fake_eval/fake_bench_v1.yaml
    uv run qf eval run --config data/processed/fake_eval/fake_bench_v1_gold.yaml
    uv run qf eval compare runs/<run of the first> runs/<run of the second>

`fake_bench_v1` answers with the gold answers of the benchmark, spoiled in a fixed pattern
(`mistake`): timeouts, truncated JSON, `pieces: 1` invented where pieces are missing,
origin and destination swapped, code fences. Every number of its report can be checked by
hand against the counts this module prints. `fake_bench_v1_gold` answers with the gold
answers unchanged. The files stay outside `configs/eval`: they are not a model configuration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

import yaml

from qf.common import DATA_PROCESSED, DATA_SPLITS, project_root, sha256_text

DEFAULT_BENCH: Final = DATA_SPLITS / "bench_v1.jsonl"
DEFAULT_OUT: Final = DATA_PROCESSED / "fake_eval"
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
    if index % 10 == 1:
        return "code_fence", f"```json\n{gold_text}\n```"
    return "gold", gold_text


def _config(name: str, model: str, responses: Path, bench: Path, sha256: str) -> dict[str, Any]:
    return {
        "name": name,
        "backend": {"name": "fake", "model": model, "responses_file": responses.as_posix()},
        "bench": {"path": bench.as_posix(), "sha256": sha256},
        "generation": {"max_tokens": 768, "temperature": 0.0},
        "metrics": METRICS,
        "slice_metrics": SLICE_METRICS,
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
    for index, line in enumerate(lines):
        messages = json.loads(line)["messages"]
        key = sha256_text(messages[1]["content"])
        kind, spoiled[key] = mistake(index, messages[2]["content"])
        gold[key] = messages[2]["content"]
        kinds[kind] += 1
    for suffix, model, answers in (("", "gold-with-mistakes", spoiled), ("_gold", "gold", gold)):
        responses = out / f"fake_responses_{stem}{suffix}.json"
        (root / responses).write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
        config = _config(f"fake_{stem}{suffix}", model, responses, bench, sha256)
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
