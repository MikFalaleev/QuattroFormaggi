"""HF transformers as a generation backend (plan step 14, D-120): the pinned base model, loaded
4-bit as in training, optionally with a LoRA adapter of `qf train run`, answering greedily.

The model gets exactly the prompt of training: the tokenizer's chat template over the
messages with the generation prompt (never a re-tokenized text), from the same local files and
with the same `fix_mistral_regex`; batches are left-padded with the `<pad>` token of the
vocabulary (never a new token). Only the new tokens are decoded. Requests are answered in
batches of `batch_size` (the `BatchGenerationBackend` port); `latency_s` is the batch time
divided by its requests. An adapter is refused if its own manifest names another base model,
revision or quantization, or the loader of the CPU rehearsal (a tiny random model).

Registered lazily in `cli.wiring` (needs torch, transformers, peft; 4-bit loading also
bitsandbytes and CUDA).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

import torch
from pydantic import Field

from qf.common import QFError, StrictConfig, project_root, read_artifact
from qf.contracts import GenerationRequest, GenerationResult, supported_versions

__all__ = ["ADAPTER_MANIFEST", "TRAINED_LOADER", "HFLocalBackend"]

ADAPTER_MANIFEST: Final = "qf_adapter_manifest.json"
"""Written into the adapter directory by `qf train run` (base, revision, loader, quantization)."""
TRAINED_LOADER: Final = "cuda_4bit"
"""The loader of real training; `tiny_random_cpu` adapters come from the CPU rehearsal."""
PAD_TOKEN: Final = "<pad>"
_DTYPES: Final[dict[str, Any]] = {"bf16": torch.bfloat16, "fp16": torch.float16,
                                  "fp32": torch.float32}  # fmt: skip


class HFLocalConfig(StrictConfig):
    base_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_dir: Path  # tokenizer, config and weights, relative to the project root
    fix_mistral_regex: bool = True  # as the tokenizer of training (BaseModelConfig, D-113)
    adapter: Path | None = None  # `runs/<id>/adapter` of `qf train run`; None: the base alone
    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"
    device: Literal["cuda", "cpu"] = "cuda"
    batch_size: int = Field(default=16, ge=1)
    allow_test_adapter: bool = False  # tests only: accept an adapter of the CPU rehearsal

    def quantization(self) -> str:
        """As `qf train run` writes it into the adapter manifest."""
        if not self.load_in_4bit:
            return f"none ({self.compute_dtype})"
        return f"{self.quant_type}, double quant {self.double_quant} (bitsandbytes)"


def _prompt_ids(tok: Any, req: GenerationRequest) -> list[int]:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = result["input_ids"] if hasattr(result, "keys") else result
    return [int(i) for i in ids]


def _refusal(req: GenerationRequest) -> str | None:
    if req.json_schema is not None:
        return "hf_local has no constrained decoding: json_schema is not supported"
    if req.temperature > 0:
        return f"hf_local decodes greedily only (temperature 0), got {req.temperature}"
    if req.messages[-1].role != "user":
        return f"the request must end with a user message, not {req.messages[-1].role}"
    return None


class HFLocalBackend:
    """`GenerationBackend` and `BatchGenerationBackend` over a local HF model."""

    Config = HFLocalConfig

    def __init__(self, cfg: HFLocalConfig) -> None:
        self.cfg = cfg
        root = project_root()
        base_dir = root / cfg.base_dir
        self._adapter_sha: str | None = None
        adapter_dir = None
        if cfg.adapter is not None:
            ref = read_artifact(cfg.adapter, "lora_adapter", supported_versions("lora_adapter"),
                                root=root)  # fmt: skip
            adapter_dir = root / ref.path
            self._check_adapter(adapter_dir)
            self._adapter_sha = ref.sha256
        self._tok = self._load_tokenizer(base_dir)
        self._model = self._load_model(base_dir, adapter_dir)

    # --- loading -----------------------------------------------------------------------------

    def _check_adapter(self, adapter_dir: Path) -> None:
        try:
            manifest = json.loads((adapter_dir / ADAPTER_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise QFError(f"{adapter_dir}: no readable {ADAPTER_MANIFEST} ({exc})") from exc
        expected = {"base_model": self.cfg.base_id, "revision": self.cfg.revision,
                    "quantization": self.cfg.quantization()}  # fmt: skip
        wrong = [f"{key} {manifest.get(key)!r} (expected {value!r})"
                 for key, value in expected.items() if manifest.get(key) != value]  # fmt: skip
        loader = manifest.get("loader")
        if loader != TRAINED_LOADER and not self.cfg.allow_test_adapter:
            wrong.append(f"loader {loader!r}: not a training run on the base model")
        if wrong:
            raise QFError(f"{adapter_dir}: adapter refused: {'; '.join(wrong)}")

    def _load_tokenizer(self, base_dir: Path) -> Any:
        from transformers import AutoTokenizer

        extra = {"fix_mistral_regex": True} if self.cfg.fix_mistral_regex else {}
        try:
            tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True, **extra)
        except Exception as exc:
            raise QFError(f"cannot load the tokenizer from {base_dir}: {exc}") from exc
        if tok.eos_token_id is None:
            raise QFError(f"{base_dir}: the tokenizer has no EOS token")
        if tok.pad_token_id is None:  # the same choice as training: an existing token only
            pad_id = tok.convert_tokens_to_ids(PAD_TOKEN)
            if not isinstance(pad_id, int) or pad_id == tok.unk_token_id:
                raise QFError(f"{base_dir}: no pad token and no {PAD_TOKEN} in the vocabulary")
            tok.pad_token = PAD_TOKEN
        tok.padding_side = "left"
        return tok

    def _load_model(self, base_dir: Path, adapter_dir: Path | None) -> Any:
        from transformers import AutoModelForCausalLM

        dtype = _DTYPES[self.cfg.compute_dtype]
        common: dict[str, Any] = {"local_files_only": True, "dtype": dtype,
                                  "attn_implementation": self.cfg.attn_implementation}  # fmt: skip
        if self.cfg.load_in_4bit:
            if self.cfg.device != "cuda" or not torch.cuda.is_available():
                raise QFError("4-bit loading needs CUDA (bitsandbytes)")
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True, bnb_4bit_quant_type=self.cfg.quant_type,
                bnb_4bit_use_double_quant=self.cfg.double_quant, bnb_4bit_compute_dtype=dtype,
            )  # fmt: skip
            model: Any = AutoModelForCausalLM.from_pretrained(
                str(base_dir), quantization_config=quantization, device_map={"": 0}, **common,
            )  # fmt: skip
        else:
            if self.cfg.device == "cuda" and not torch.cuda.is_available():
                raise QFError("device cuda requested, but CUDA is not available")
            model = AutoModelForCausalLM.from_pretrained(str(base_dir), **common)
            model = model.to(self.cfg.device)
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.eval()
        return model

    # --- the ports -----------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "hf_local"

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    def model_id(self) -> str:
        precision = (f"4bit-{self.cfg.quant_type}{'-dq' if self.cfg.double_quant else ''}"
                     f"-{self.cfg.compute_dtype}" if self.cfg.load_in_4bit
                     else self.cfg.compute_dtype)  # fmt: skip
        model = f"{self.cfg.base_id}@{self.cfg.revision[:12]} {precision}"
        return model if self._adapter_sha is None else f"{model} + lora {self._adapter_sha[:12]}"

    def generate(self, req: GenerationRequest) -> GenerationResult:
        return self.generate_batch([req])[0]

    def generate_batch(self, reqs: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results: list[GenerationResult | None] = [None] * len(reqs)
        groups: dict[int, list[tuple[int, list[int]]]] = {}  # max_tokens -> (index, prompt)
        for i, req in enumerate(reqs):
            problem = _refusal(req)
            if problem is not None:
                results[i] = GenerationResult(text="", latency_s=0.0, error=problem)
                continue
            try:
                groups.setdefault(req.max_tokens, []).append((i, _prompt_ids(self._tok, req)))
            except Exception as exc:  # a template that cannot render these messages
                error = f"chat template: {type(exc).__name__}: {exc}"
                results[i] = GenerationResult(text="", latency_s=0.0, error=error)
        for max_tokens, items in groups.items():
            started = time.monotonic()
            try:
                outputs = self._generate_ids([ids for _, ids in items], max_tokens)
            except Exception as exc:  # out of memory and the like: every request of the batch
                latency = (time.monotonic() - started) / len(items)
                for i, _ in items:
                    results[i] = GenerationResult(text="", latency_s=latency,
                                                  error=f"{type(exc).__name__}: {exc}")  # fmt: skip
                continue
            latency = (time.monotonic() - started) / len(items)
            for (i, ids), new in zip(items, outputs, strict=True):
                text = str(self._tok.decode(new, skip_special_tokens=True)).strip()
                results[i] = GenerationResult(text=text, latency_s=latency, prompt_tokens=len(ids),
                                              completion_tokens=len(new))  # fmt: skip
        return [r for r in results if r is not None]

    def _generate_ids(self, prompts: Sequence[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Greedy continuations of left-padded prompts: the new tokens of each, up to and
        including EOS (padding after EOS removed)."""
        from transformers import GenerationConfig

        pad_id, eos_id = int(self._tok.pad_token_id), int(self._tok.eos_token_id)
        width = max(len(ids) for ids in prompts)
        input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, ids in enumerate(prompts):
            input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention[row, width - len(ids) :] = 1
        device = next(self._model.parameters()).device
        settings = GenerationConfig(  # type: ignore[no-untyped-call]
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
            eos_token_id=eos_id, use_cache=True,
        )  # fmt: skip
        with torch.inference_mode():
            out = self._model.generate(input_ids=input_ids.to(device),
                                       attention_mask=attention.to(device),
                                       generation_config=settings)  # fmt: skip
        continuations = []
        for row in out[:, width:].tolist():
            end = row.index(eos_id) + 1 if eos_id in row else len(row)
            continuations.append([int(t) for t in row[:end]])
        return continuations
