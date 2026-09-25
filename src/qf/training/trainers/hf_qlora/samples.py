"""Sample answers right after training (D-118) — NOT an evaluation.

A few val records (never the benchmark) answered greedily by the model with the adapter and,
for contrast, with the adapter disabled. It shows whether the model learnt the answer format
before the GPU machine is deleted; quality is measured by `qf eval` (plan step 14).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final

import torch

from qf.contracts import SFTRecord
from qf.training import chat_ids

__all__ = ["SAMPLES_FILE", "generate_samples", "render_samples"]

SAMPLES_FILE: Final = "samples.md"


def _json_ok(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _answer(model: Any, tok: Any, prompt: list[int], max_new_tokens: int, pad_id: int) -> str:
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=pad_id)  # fmt: skip
    return str(tok.decode(out[0, len(prompt) :], skip_special_tokens=True)).strip()


def generate_samples(model: Any, tok: Any, records: Sequence[SFTRecord], *,
                     max_new_tokens: int, pad_id: int) -> list[dict[str, Any]]:  # fmt: skip
    """Greedy answers with and without the adapter; the KV cache is on and gradient
    checkpointing off (otherwise every new token recomputes the whole prompt)."""
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    samples = []
    for record in records:
        messages = [{"role": m.role, "content": m.content} for m in record.messages[:2]]
        prompt = chat_ids(tok, messages, add_generation_prompt=True)
        adapter = _answer(model, tok, prompt, max_new_tokens, pad_id)
        with model.disable_adapter():
            base = _answer(model, tok, prompt, max_new_tokens, pad_id)
        samples.append({"id": record.id, "expected": record.messages[2].content,
                        "adapter": adapter, "adapter_json": _json_ok(adapter),
                        "base": base, "base_json": _json_ok(base)})  # fmt: skip
    model.config.use_cache = False
    return samples


def render_samples(samples: Sequence[dict[str, Any]], run_id: str) -> str:
    lines = [f"# Образцы ответов — не оценка ({run_id})", "",
             "Несколько записей из val, жадная генерация. Показывают, выучен ли формат ответа; "
             "качество измеряет `qf eval` (шаг 14).", ""]  # fmt: skip

    def parses(ok: bool) -> str:
        return "да" if ok else "нет"

    for sample in samples:
        lines += [f"## {sample['id']}", "",
                  f"**С адаптером** (JSON разбирается: {parses(sample['adapter_json'])}):",
                  "", "```", sample["adapter"], "```", "",
                  f"**Без адаптера** (JSON разбирается: {parses(sample['base_json'])}):",
                  "", "```", sample["base"], "```", "", "**Эталон:**", "", "```",
                  sample["expected"], "```", ""]  # fmt: skip
    return "\n".join(lines)
