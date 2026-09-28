"""Checks of a merged model (plan step 16, D-122).

`verify_merged_files` — files only (any machine, e.g. the Mac copy): the directory hash against
its artifact manifest, every file against `qf_export_manifest.json`, the tokenizer files equal
to the base's. `verify_merged` — with the models: on fixed prompts the merged model and base +
adapter give the same greedy tokens (the pass rule), their last-token logits are close
(`max_abs_diff` recorded, no tolerance at 12B in bf16), and both differ from the base alone
(the adapter disabled on the same model): a merge that changed nothing does not pass. The two
models are loaded one after the other, never together. torch and peft are imported inside
the functions.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qf.common import QFError, read_artifact, sha256_file
from qf.contracts import supported_versions
from qf.export.config import EXPORT_MANIFEST

__all__ = ["MergeVerification", "PromptCheck", "verify_merged", "verify_merged_files"]


@dataclass
class PromptCheck:
    greedy_equal: bool  # merged == base + adapter on every generated token
    differs_from_base: bool  # the base alone generates other tokens
    max_abs_diff: float  # last-token logits, merged vs base + adapter
    base_max_abs_diff: float  # last-token logits, merged vs the base alone
    first_divergence: int | None = None  # the first generated position where merged != adapter
    margin_at_divergence: float | None = None  # top-1 minus top-2 logit of base + adapter there


@dataclass
class MergeVerification:
    path: str
    sha256: str | None = None
    prompts: list[PromptCheck] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "ok": self.ok, "checks": self.checks,
                "problems": self.problems, "prompts": [vars(p) for p in self.prompts]}  # fmt: skip


def verify_merged_files(merged: Path, root: Path, *, base_dir: Path | None = None,
                        result: MergeVerification | None = None) -> MergeVerification:  # fmt: skip
    """The merged directory as files: artifact hash, the file list of its own manifest, the
    tokenizer files byte-equal to the base's (when `base_dir` is on this machine)."""
    result = result or MergeVerification(path=str(merged))
    try:
        ref = read_artifact(merged, "hf_model", supported_versions("hf_model"), root=root)
    except QFError as exc:
        result.problems.append(str(exc))
        return result
    result.sha256 = ref.sha256
    result.checks.append(f"sha256 {ref.sha256[:12]}… matches its manifest")
    directory = root / ref.path
    try:
        manifest = json.loads((directory / EXPORT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result.problems.append(f"{EXPORT_MANIFEST}: not readable ({exc})")
        return result
    listed: dict[str, str] = manifest.get("files", {})
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
               if p.is_file() and p.name != EXPORT_MANIFEST}  # fmt: skip
    wrong = sorted(n for n, sha in listed.items()
                   if not (directory / n).is_file()
                   or sha256_file(directory / n) != sha)  # fmt: skip
    if wrong or present != set(listed):
        result.problems.append(f"files differ from {EXPORT_MANIFEST}: changed {wrong[:5]}, "
                               f"unlisted {sorted(present - set(listed))[:5]}")  # fmt: skip
    else:
        result.checks.append(f"{len(listed)} files match {EXPORT_MANIFEST}")
    if base_dir is not None and base_dir.is_dir():
        copied = manifest.get("copied_from_base", [])
        differ = [n for n in copied if not (base_dir / n).is_file()
                  or sha256_file(base_dir / n) != listed.get(n)]  # fmt: skip
        if differ:
            result.problems.append(f"tokenizer files differ from the base: {differ}")
        else:
            result.checks.append(f"tokenizer and generation files equal to the base: {copied}")
    return result


def _generate(model: Any, prompts: Sequence[list[int]], max_new_tokens: int,
              pad_id: int, eos_id: int) -> list[tuple[Any, list[int], list[float]]]:  # fmt: skip
    """Per prompt: last-token logits (float32, CPU), greedy new tokens, and the top-1 minus
    top-2 logit margin at each generated position."""
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for ids in prompts:
            tensor = torch.tensor([ids], device=device)
            logits = model(input_ids=tensor).logits[0, -1].float().cpu()
            generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                       do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=pad_id, eos_token_id=eos_id,
                                       output_scores=True,
                                       return_dict_in_generate=True)  # fmt: skip
            margins = []
            for step in generated.scores:
                top = torch.topk(step[0].float(), 2).values
                margins.append(float(top[0] - top[1]))
            new = [int(t) for t in generated.sequences[0, len(ids) :].tolist()]
            out.append((logits, new, margins))
    return out


def _divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _free_memory() -> None:
    """After the caller dropped its model: return the memory before loading the next one."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_merged(merged: Path, adapter: Path, base_dir: Path, root: Path, *,
                  messages: Sequence[Sequence[dict[str, str]]], max_new_tokens: int,
                  dtype: str, device: str,
                  fix_mistral_regex: bool = False) -> MergeVerification:  # fmt: skip
    """Files first, then base + adapter (and the base alone) against the merged model."""
    os.environ["HF_HUB_OFFLINE"] = "1"  # before peft/transformers import the Hub client
    from peft import PeftModel
    from transformers import AutoTokenizer

    from qf.export.mergers.peft_merge import load_causal_lm

    result = verify_merged_files(merged, root, base_dir=root / base_dir)
    if not result.ok:
        return result
    extra = {"fix_mistral_regex": True} if fix_mistral_regex else {}
    tok: Any = AutoTokenizer.from_pretrained(str(root / merged), local_files_only=True, **extra)
    pad = tok.pad_token_id
    pad_id = int(pad if pad is not None else tok.convert_tokens_to_ids("<pad>"))
    prompts = []
    for chat in messages:
        rendered: Any = tok.apply_chat_template(list(chat), tokenize=True,
                                                add_generation_prompt=True)  # fmt: skip
        ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
        prompts.append([int(i) for i in ids])
    tuned = PeftModel.from_pretrained(load_causal_lm(root / base_dir, dtype, device),
                                      str(root / adapter), is_trainable=False)  # fmt: skip
    tuned.eval()
    with_adapter = _generate(tuned, prompts, max_new_tokens, pad_id, tok.eos_token_id)
    with tuned.disable_adapter():
        base_alone = _generate(tuned, prompts, max_new_tokens, pad_id, tok.eos_token_id)
    del tuned
    _free_memory()
    model = load_causal_lm(root / merged, dtype, device)
    model.eval()
    merged_out = _generate(model, prompts, max_new_tokens, pad_id, tok.eos_token_id)
    del model
    _free_memory()
    for (a_logits, a_ids, a_margins), (b_logits, b_ids, _), (m_logits, m_ids, _) in zip(
            with_adapter, base_alone, merged_out, strict=True):  # fmt: skip
        split = _divergence(m_ids, a_ids)
        result.prompts.append(PromptCheck(
            greedy_equal=m_ids == a_ids, differs_from_base=m_ids != b_ids,
            max_abs_diff=float((m_logits - a_logits).abs().max()),
            base_max_abs_diff=float((m_logits - b_logits).abs().max()),
            first_divergence=split,
            margin_at_divergence=a_margins[split] if split is not None
            and split < len(a_margins) else None))  # fmt: skip
    equal = sum(p.greedy_equal for p in result.prompts)
    differs = sum(p.differs_from_base for p in result.prompts)
    worst = max(p.max_abs_diff for p in result.prompts)
    if equal != len(prompts):
        where = [(p.first_divergence, p.margin_at_divergence) for p in result.prompts
                 if not p.greedy_equal]  # fmt: skip
        result.problems.append(f"greedy tokens of the merged model differ from base + adapter "
                               f"on {len(prompts) - equal} of {len(prompts)} prompts; (first "
                               f"diverging position, top-2 margin there): {where}")  # fmt: skip
    else:
        result.checks.append(f"greedy {max_new_tokens} tokens equal to base + adapter on "
                             f"{equal} of {len(prompts)} prompts; max |Δ logit| "
                             f"{worst:.4g}")  # fmt: skip
    if differs == 0:
        result.problems.append("the merged model answers as the base alone: the merge changed "
                               "nothing")  # fmt: skip
    else:
        result.checks.append(f"differs from the base alone on {differs} of {len(prompts)} prompts")
    return result
