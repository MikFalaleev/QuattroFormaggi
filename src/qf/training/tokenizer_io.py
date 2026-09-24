"""The tokenizer of the base model (plan step 11): the pinned files, their hashes, loading.

Only the tokenizer and config files are downloaded, never weights. `transformers` is imported
inside the functions that need it (plan C.3): importing this module stays light.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import Field

from qf.common import QFError, StrictConfig, atomic_write_text, sha256_file, sha256_text

__all__ = [
    "PROVENANCE_FILE",
    "BaseModelConfig",
    "chat_template_hash",
    "download_tokenizer_files",
    "load_tokenizer",
    "tokenizer_hash",
    "verify_tokenizer_files",
]

PROVENANCE_FILE: Final = "provenance.json"
_HASHED: Final = ("tokenizer.json", "tokenizer_config.json")


class BaseModelConfig(StrictConfig):
    """`configs/train/base_model.yaml`: the pinned base model of QF-12B."""

    id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # a full commit sha, never a branch name
    license: str
    tokenizer_files: list[str] = Field(min_length=1)
    local_dir: Path  # relative to the project root, outside Git
    # transformers 5 warns that the pre-tokenizer regex of this tokenizer.json differs from the
    # official Mistral tokenizer; on our data both give the same ids (D-113), the fix is kept on
    fix_mistral_regex: bool = True

    def directory(self, root: Path) -> Path:
        return root / self.local_dir


def download_tokenizer_files(cfg: BaseModelConfig, root: Path) -> Path:
    """The tokenizer and config files of the pinned revision into `cfg.local_dir`, with a
    provenance file (sha256 of each file). Nothing else is downloaded."""
    from huggingface_hub import hf_hub_download

    dest = cfg.directory(root)
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in cfg.tokenizer_files:
        try:
            path = Path(hf_hub_download(repo_id=cfg.id, filename=name, revision=cfg.revision,
                                        local_dir=dest))  # fmt: skip
        except Exception as exc:  # network, 404, gated repository
            raise QFError(f"cannot download {cfg.id}@{cfg.revision[:12]}/{name}: {exc}") from exc
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    provenance = {"model": cfg.id, "revision": cfg.revision, "license": cfg.license,
                  "downloaded_at": datetime.now(UTC).isoformat(), "files": files}  # fmt: skip
    atomic_write_text(dest / PROVENANCE_FILE, json.dumps(provenance, indent=2) + "\n")
    return dest


def verify_tokenizer_files(cfg: BaseModelConfig, root: Path) -> dict[str, Any]:
    """The provenance of the downloaded files, after checking that every file is still there
    with the recorded sha256 and belongs to the configured revision."""
    dest = cfg.directory(root)
    try:
        provenance: dict[str, Any] = json.loads((dest / PROVENANCE_FILE).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise QFError(f"no tokenizer files in {dest}: run `qf tokens fetch` ({exc})") from exc
    if (provenance.get("model"), provenance.get("revision")) != (cfg.id, cfg.revision):
        raise QFError(f"{dest} holds {provenance.get('model')}@{provenance.get('revision')}, "
                      f"not {cfg.id}@{cfg.revision}")  # fmt: skip
    for name in cfg.tokenizer_files:
        expected = provenance["files"].get(name, {}).get("sha256")
        if not (dest / name).exists() or sha256_file(dest / name) != expected:
            raise QFError(f"{dest / name} is missing or changed since download")
    return provenance


def tokenizer_hash(directory: Path) -> str:
    """sha256 of `tokenizer.json` and `tokenizer_config.json` (their hashes, in this order)."""
    return sha256_text("\n".join(sha256_file(directory / name) for name in _HASHED))


def chat_template_hash(tok: Any) -> str:
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise QFError("the tokenizer has no chat template")
    return sha256_text(template)


def load_tokenizer(directory: Path, *, fix_mistral_regex: bool = False) -> Any:
    """The fast tokenizer with its chat template, from the local files only."""
    from transformers import AutoTokenizer

    extra = {"fix_mistral_regex": True} if fix_mistral_regex else {}
    try:
        return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, **extra)
    except Exception as exc:
        raise QFError(f"cannot load the tokenizer from {directory}: {exc}") from exc
