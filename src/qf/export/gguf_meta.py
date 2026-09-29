"""GGUF metadata and its check against the HF model it was made from (plan step 17, D-123).

`read_gguf_metadata` uses the `gguf` package of the pinned llama.cpp checkout (`gguf-py`, only
numpy needed) and returns the keys the check needs; `check_gguf_against_hf` compares the chat
template (sha256), BOS and EOS ids with the HF tokenizer files. Pure files: no model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Final

from qf.common import QFError

__all__ = ["KEYS", "check_gguf_against_hf", "hf_tokenizer_facts", "read_gguf_metadata"]

KEYS: Final = (
    "general.architecture", "general.name", "general.file_type", "tokenizer.ggml.model",
    "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.chat_template",
)  # fmt: skip


def _sha(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_gguf_metadata(path: Path, gguf_py: Path) -> dict[str, Any]:
    """`KEYS` present in the file, the vocabulary size and the chat template's sha256."""
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise QFError(f"gguf-py not found in {gguf_py} (llama.cpp checkout)") from exc
    reader = GGUFReader(str(path))
    meta: dict[str, Any] = {}
    for key in KEYS:
        field = reader.fields.get(key)
        if field is not None:
            meta[key] = field.contents()
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    meta["vocab_size"] = len(tokens.data) if tokens is not None else None
    meta["chat_template_sha256"] = _sha(meta.get("tokenizer.chat_template"))
    return meta


def hf_tokenizer_facts(hf_dir: Path) -> dict[str, Any]:
    """Chat template sha256 and BOS/EOS ids of an HF model directory (tokenizer files only)."""
    config = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    jinja = hf_dir / "chat_template.jinja"
    if template is None and jinja.is_file():
        template = jinja.read_text(encoding="utf-8")
    tokenizer = json.loads((hf_dir / "tokenizer.json").read_text(encoding="utf-8"))
    added = tokenizer.get("added_tokens", [])
    ids = {t["content"]: t["id"] for t in added}

    def token_id(name: str) -> int | None:
        token = config.get(name)
        content = token.get("content") if isinstance(token, dict) else token
        return ids.get(content) if content is not None else None

    return {"chat_template_sha256": _sha(template), "bos_token_id": token_id("bos_token"),
            "eos_token_id": token_id("eos_token")}  # fmt: skip


def check_gguf_against_hf(meta: dict[str, Any], hf_dir: Path) -> list[str]:
    """Problems: the chat template, BOS or EOS of the GGUF differ from the HF model."""
    hf = hf_tokenizer_facts(hf_dir)
    problems = []
    if meta.get("chat_template_sha256") != hf["chat_template_sha256"]:
        problems.append(f"chat template sha256 {str(meta.get('chat_template_sha256'))[:12]} ≠ HF "
                        f"{str(hf['chat_template_sha256'])[:12]}")  # fmt: skip
    for key in ("bos_token_id", "eos_token_id"):
        if meta.get(f"tokenizer.ggml.{key}") != hf[key]:
            problems.append(f"{key} {meta.get(f'tokenizer.ggml.{key}')} ≠ HF {hf[key]}")
    return problems
