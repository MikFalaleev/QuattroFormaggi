from __future__ import annotations

import hashlib
from pathlib import Path

from qf.common import canonical_json, sha256_file, sha256_json, sha256_text


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    data = b"quattro formaggi\n" * 100_000  # larger than one 1 MiB read chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected
    small = tmp_path / "small.bin"
    small.write_bytes(data[:1000])
    assert sha256_file(small, chunk=7) == hashlib.sha256(data[:1000]).hexdigest()


def test_sha256_json_key_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_text_utf8() -> None:
    text = "Канзас-Сити, 12,6 т"
    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_canonical_json_keeps_cyrillic_unescaped() -> None:
    assert canonical_json({"город": "Хьюстон", "a": [1, 2]}) == '{"a":[1,2],"город":"Хьюстон"}'
