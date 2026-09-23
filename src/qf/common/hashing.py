"""SHA-256 helpers for files, text and canonical JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["canonical_json", "sha256_file", "sha256_json", "sha256_text"]

_DEFAULT_CHUNK = 1 << 20


def sha256_file(path: Path, chunk: int = _DEFAULT_CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialize so that equal objects give equal strings regardless of key order."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))
