"""llama.cpp as the GGUF converter and quantizer (plan step 17, D-123).

`llama_cpp_convert` runs `convert_hf_to_gguf.py` of the pinned llama.cpp checkout with its own
venv (`.venv-convert`, the checkout's requirements); `llama_cpp_quantize` runs `llama-quantize`
built from the same checkout. Both refuse a checkout at another commit than the config's, run
the tool with `check=True`, write its stdout and stderr to `runs/<run_id>/`, raise `QFError`
with the tail of stderr on failure and write a `gguf` artifact manifest (the llama.cpp commit,
the type, the source). Paths come from `configs/export/gguf.yaml`, never constants.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import Field

from qf.common import (
    RUNS,
    ArtifactRef,
    QFError,
    StrictConfig,
    project_root,
    read_artifact,
    write_artifact,
)
from qf.contracts import supported_versions

__all__ = [
    "LlamaCppConvertConfig",
    "LlamaCppConverter",
    "LlamaCppQuantizeConfig",
    "LlamaCppQuantizer",
    "head_commit",
    "run_tool",
]

GGUF_VERSION: Final = "gguf_v3"
_TAIL: Final = 2000


class LlamaCppConfig(StrictConfig):
    llama_cpp_dir: Path = Path("third_party/llama.cpp")  # the submodule, relative to the root
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")  # the pinned commit (user's decision, D-123)


class LlamaCppConvertConfig(LlamaCppConfig):
    python: Path = Path(".venv-convert/bin/python")  # the converter's own venv


class LlamaCppQuantizeConfig(LlamaCppConfig):
    quantize_bin: Path = Path("third_party/llama.cpp-build/bin/llama-quantize")
    threads: int | None = Field(default=None, ge=1)


def head_commit(directory: Path) -> str:
    """The checked-out commit of a git working tree."""
    try:
        out = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True)  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QFError(f"{directory}: not a git checkout of llama.cpp ({exc})") from exc
    return out.stdout.strip()


def _check_commit(cfg: LlamaCppConfig, root: Path) -> None:
    head = head_commit(root / cfg.llama_cpp_dir)
    if head != cfg.commit:
        raise QFError(f"{cfg.llama_cpp_dir} is at {head}, the config pins {cfg.commit}")


def run_tool(args: Sequence[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    """Run an external tool; its output goes to `log`; failure -> `QFError` with stderr tail."""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        # tool logs are not always UTF-8 (llama-quantize prints raw token bytes): never fail on them
        result = subprocess.run(list(args), check=True, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env=env)  # fmt: skip
    except subprocess.CalledProcessError as exc:
        log.write_text(f"$ {' '.join(args)}\n{exc.stdout or ''}\n--- stderr ---\n"
                       f"{exc.stderr or ''}", encoding="utf-8")  # fmt: skip
        tail = (exc.stderr or "")[-_TAIL:]
        raise QFError(f"{Path(args[1] if len(args) > 1 else args[0]).name} failed with exit "
                      f"code {exc.returncode}:\n{tail}") from exc  # fmt: skip
    except OSError as exc:
        raise QFError(f"cannot run {args[0]}: {exc}") from exc
    log.write_text(f"$ {' '.join(args)}\n{result.stdout}\n--- stderr ---\n{result.stderr}",
                   encoding="utf-8")  # fmt: skip


def _new_output(out: Path, root: Path) -> Path:
    absolute = root / out
    if absolute.exists():
        raise QFError(f"{out} already exists: remove it or choose another name")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


class LlamaCppConverter:
    """`ModelConverter`: HF directory -> GGUF via `convert_hf_to_gguf.py`."""

    Config = LlamaCppConvertConfig

    def __init__(self, cfg: LlamaCppConvertConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "llama_cpp_convert"

    def convert(self, source: Path, out: Path, *, outtype: str, run_id: str,
                parents: Sequence[str]) -> ArtifactRef:  # fmt: skip
        root = project_root()
        _check_commit(self.cfg, root)
        target = _new_output(out, root)
        args = [str(root / self.cfg.python),
                str(root / self.cfg.llama_cpp_dir / "convert_hf_to_gguf.py"),
                str(root / source), "--outtype", outtype.lower(),
                "--outfile", str(target)]  # fmt: skip
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        run_tool(args, root / RUNS / run_id / f"convert-{out.name}.log", env=env)
        return write_artifact(target, "gguf", GGUF_VERSION, run_id, parents,
                              extra={"llama_cpp_commit": self.cfg.commit, "tool": self.name,
                                     "type": outtype.upper(), "hf_source": source.as_posix()},
                              root=root)  # fmt: skip


class LlamaCppQuantizer:
    """`Quantizer`: GGUF -> quantized GGUF via `llama-quantize`."""

    Config = LlamaCppQuantizeConfig

    def __init__(self, cfg: LlamaCppQuantizeConfig) -> None:
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "llama_cpp_quantize"

    def quantize(self, source: ArtifactRef, out: Path, *, qtype: str,
                 run_id: str) -> ArtifactRef:  # fmt: skip
        from qf.common import load_artifact_manifest

        root = project_root()
        _check_commit(self.cfg, root)
        read_artifact(Path(source.path), "gguf", supported_versions("gguf"), root=root)
        target = _new_output(out, root)
        args = [str(root / self.cfg.quantize_bin), str(root / source.path), str(target), qtype]
        if self.cfg.threads is not None:
            args.append(str(self.cfg.threads))
        run_tool(args, root / RUNS / run_id / f"quantize-{out.name}.log")
        hf_source = load_artifact_manifest(root / source.path, root=root).extra.get("hf_source")
        return write_artifact(target, "gguf", GGUF_VERSION, run_id, [source.sha256],
                              extra={"llama_cpp_commit": self.cfg.commit, "tool": self.name,
                                     "type": qtype, "gguf_source": source.path.as_posix(),
                                     "hf_source": hf_source},
                              root=root)  # fmt: skip
