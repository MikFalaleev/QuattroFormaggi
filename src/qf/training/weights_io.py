"""The weights of the pinned base model (fetched on the GPU machine at plan step 13, ⛔ STOP).

Only the files listed in `weight_files` are downloaded: the HF-format shards and their index.
Each file is checked against the size and LFS sha256 the Hub reports for the pinned revision.
The training loader reads the weights from this directory with `local_files_only`, so nothing
is ever downloaded implicitly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from qf.common import QFError, atomic_write_text, sha256_file
from qf.training.tokenizer_io import BaseModelConfig

__all__ = ["WEIGHTS_PROVENANCE_FILE", "download_base_weights", "hub_files", "verify_base_weights"]

WEIGHTS_PROVENANCE_FILE: Final = "weights_provenance.json"


def hub_files(cfg: BaseModelConfig) -> dict[str, dict[str, Any]]:
    """{file: {"bytes", "sha256" (LFS files only)}} of the pinned revision on the Hub."""
    from huggingface_hub import HfApi

    try:
        entries = HfApi().list_repo_tree(cfg.id, revision=cfg.revision, recursive=True)
        files = {}
        for entry in entries:
            lfs = getattr(entry, "lfs", None)
            files[entry.path] = {"bytes": getattr(entry, "size", None),
                                 "sha256": getattr(lfs, "sha256", None)}  # fmt: skip
    except Exception as exc:  # network, 404
        raise QFError(f"cannot list {cfg.id}@{cfg.revision[:12]}: {exc}") from exc
    return files


def download_base_weights(cfg: BaseModelConfig, root: Path) -> Path:
    """The `weight_files` of the pinned revision into `cfg.local_dir`, each checked against the
    Hub's size and sha256, with a provenance file."""
    from huggingface_hub import hf_hub_download

    if not cfg.weight_files:
        raise QFError("no weight_files in the base model config")
    listed = hub_files(cfg)
    missing = [name for name in cfg.weight_files if name not in listed]
    if missing:
        raise QFError(f"{cfg.id}@{cfg.revision[:12]} has no {missing}")
    dest = cfg.directory(root)
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in cfg.weight_files:
        try:
            path = Path(hf_hub_download(repo_id=cfg.id, filename=name, revision=cfg.revision,
                                        local_dir=dest))  # fmt: skip
        except Exception as exc:
            raise QFError(f"cannot download {cfg.id}@{cfg.revision[:12]}/{name}: {exc}") from exc
        size, digest = path.stat().st_size, sha256_file(path)
        expected = listed[name]
        if expected["bytes"] not in (None, size) or expected["sha256"] not in (None, digest):
            raise QFError(f"{name}: downloaded {size} bytes, sha256 {digest[:12]}…; the Hub "
                          f"reports {expected['bytes']} bytes, "
                          f"{str(expected['sha256'])[:12]}…")  # fmt: skip
        files[name] = {"sha256": digest, "bytes": size}
    provenance = {"model": cfg.id, "revision": cfg.revision, "license": cfg.license,
                  "downloaded_at": datetime.now(UTC).isoformat(), "files": files}  # fmt: skip
    atomic_write_text(dest / WEIGHTS_PROVENANCE_FILE, json.dumps(provenance, indent=2) + "\n")
    return dest


def verify_base_weights(cfg: BaseModelConfig, root: Path, *, full_hash: bool = False
                        ) -> dict[str, Any]:  # fmt: skip
    """The weights provenance after checking model, revision and every file's size (and sha256
    with `full_hash`, which reads ~24 GB)."""
    dest = cfg.directory(root)
    try:
        provenance: dict[str, Any] = json.loads((dest / WEIGHTS_PROVENANCE_FILE).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise QFError(
            f"no base model weights in {dest}: run `qf train fetch-base` (a ~24.5 GB "
            f"download that needs the user's approval, plan step 13)"
        ) from exc
    if (provenance.get("model"), provenance.get("revision")) != (cfg.id, cfg.revision):
        raise QFError(f"{dest} holds weights of {provenance.get('model')}@"
                      f"{provenance.get('revision')}, not {cfg.id}@{cfg.revision}")  # fmt: skip
    for name in cfg.weight_files:
        recorded = provenance["files"].get(name)
        path = dest / name
        if recorded is None or not path.exists() or path.stat().st_size != recorded["bytes"]:
            raise QFError(f"{path} is missing or changed since download")
        if full_hash and sha256_file(path) != recorded["sha256"]:
            raise QFError(f"{path}: sha256 differs from the download")
    return provenance
