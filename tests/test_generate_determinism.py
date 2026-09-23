from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys

from qf.data import generate_records
from tests.conftest import REPO_ROOT
from tests.generation import SOURCE_PREFIX, repo_config, synthetic_facts

POOL = {"train": 60, "val": 10, "test": 10, "test_ood": 15, "smoke": 5}
OOD = {"holdout_route_count": 2, "holdout_families": ["T7", "T8"]}

_SCRIPT = """
import hashlib
from tests.generation import SOURCE_PREFIX, repo_config, synthetic_facts
from qf.data import generate_records
cfg = repo_config(pool={pool!r}, ood={ood!r})
records, _ = generate_records(synthetic_facts(), cfg, source_prefix=SOURCE_PREFIX)
text = "".join(r.model_dump_json() + "\\n" for items in records.values() for r in items)
print(hashlib.sha256(text.encode()).hexdigest())
"""


def _digest(seed: int = 42, shuffle: bool = False) -> str:
    facts = synthetic_facts()
    if shuffle:
        random.Random(5).shuffle(facts)
    cfg = repo_config(pool=POOL, ood=OOD, seed=seed)
    records, _ = generate_records(facts, cfg, source_prefix=SOURCE_PREFIX)
    text = "".join(r.model_dump_json() + "\n" for items in records.values() for r in items)
    return hashlib.sha256(text.encode()).hexdigest()


def test_same_seed_same_bytes() -> None:
    assert _digest() == _digest()


def test_same_bytes_across_processes_with_different_hash_seeds() -> None:
    """Set iteration order depends on PYTHONHASHSEED; the output must not."""
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT.format(pool=POOL, ood=OOD)],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
        )  # fmt: skip
        digests.add(result.stdout.strip())
    assert digests == {_digest()}


def test_different_seed_differs() -> None:
    assert _digest(seed=42) != _digest(seed=43)


def test_record_independent_of_order() -> None:
    assert _digest(shuffle=True) == _digest()
