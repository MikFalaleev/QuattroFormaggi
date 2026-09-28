"""A tiny random Mistral with the word-level tokenizer of `tests.tiny_training`, saved as a local
base model in a fake project root, and a LoRA adapter for it with the manifests of `qf train
run`: CPU tests of the `hf_local` backend (step 14). Nothing is downloaded.

Needs torch, transformers and peft (import this module after importorskip).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import MistralConfig, MistralForCausalLM

from qf.common import write_artifact
from qf.contracts import GenerationRequest, Message
from tests.tiny_training import REVISION, SYSTEM, tiny_tokenizer, user_text

TINY_BASE = Path("artifacts/base_model/fake-tiny-hf")
TINY_ID = "fake/tiny"
ADAPTER = Path("runs/20260101-000000-train-tiny00/adapter")


def tiny_base(root: Path) -> Path:
    directory = root / TINY_BASE
    tok = tiny_tokenizer()
    tok.save_pretrained(directory)
    torch.manual_seed(0)
    config = MistralConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=256, initializer_range=0.1,
                           tie_word_embeddings=False, bos_token_id=tok.bos_token_id,
                           eos_token_id=tok.eos_token_id)  # fmt: skip
    MistralForCausalLM(config).save_pretrained(directory)
    return TINY_BASE


def tiny_adapter(root: Path, *, loader: str = "tiny_random_cpu", zero: bool = False,
                 **manifest: Any) -> Path:  # fmt: skip
    """A LoRA adapter with non-zero weights (it changes the answers; `zero`: PEFT's initial
    all-zero `lora_B`) and its manifests, named as `qf train run` names them."""
    model = MistralForCausalLM.from_pretrained(root / TINY_BASE)
    torch.manual_seed(1)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                      init_lora_weights=zero)  # fmt: skip
    directory = root / ADAPTER
    get_peft_model(model, lora).save_pretrained(directory)
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config |= {"base_model_name_or_path": TINY_ID, "revision": REVISION}  # as name_base_model
    config_path.write_text(json.dumps(config), encoding="utf-8")
    own = {"base_model": TINY_ID, "revision": REVISION, "loader": loader,
           "quantization": "none (fp32)", "run_id": directory.parent.name,
           "experiment_id": None, "seed": 1, "global_step": 1, **manifest}  # fmt: skip
    (directory / "qf_adapter_manifest.json").write_text(json.dumps(own), encoding="utf-8")
    write_artifact(directory, "lora_adapter", "peft_lora_v1", directory.parent.name, root=root)
    return ADAPTER


def hf_config(**changes: Any) -> dict[str, Any]:
    """`hf_local` parameters for the tiny base model on the CPU in fp32."""
    return {"base_id": TINY_ID, "revision": REVISION, "base_dir": TINY_BASE.as_posix(),
            "fix_mistral_regex": False, "load_in_4bit": False, "compute_dtype": "fp32",
            "device": "cpu", "batch_size": 4, **changes}  # fmt: skip


def tiny_request(i: int, *, max_tokens: int = 12, repeat: int = 1,
                 **changes: Any) -> GenerationRequest:  # fmt: skip
    """System and user messages; `repeat` makes the user message (and the prompt) longer."""
    user = " , ".join([user_text(i)] * repeat)
    messages = [Message(role="system", content=SYSTEM), Message(role="user", content=user)]
    return GenerationRequest(messages=messages, max_tokens=max_tokens, temperature=0.0,
                             **changes)  # fmt: skip
