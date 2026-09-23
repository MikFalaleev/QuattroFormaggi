from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.conftest import REPO_ROOT

HEAVY = ("torch", "transformers", "peft", "bitsandbytes", "gguf")
# Adapter modules may import heavy libraries at module level (plan C.3); they are loaded only
# through lazy registry entries, so everything else must import without them.
ADAPTER_PREFIXES = (
    "qf.backends.hf_local",
    "qf.training.trainers",
    "qf.export.mergers",
    "qf.export.converters",
)

SCRIPT = textwrap.dedent(
    f"""
    import importlib, pkgutil, sys
    for name in {HEAVY!r}:
        sys.modules[name] = None  # any import of it now raises ImportError
    import qf
    imported = []
    for info in pkgutil.walk_packages(qf.__path__, "qf."):
        if info.name == "qf.__main__" or info.name.startswith({ADAPTER_PREFIXES!r}):
            continue
        importlib.import_module(info.name)
        imported.append(info.name)
    import qf.cli.wiring
    print(len(imported))
    """
)


def test_light_packages_import_without_heavy_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= 20  # every non-adapter module was actually imported


def test_heavy_blocking_really_blocks() -> None:
    """Guard against a vacuous pass: the blocking trick must make `import torch` fail."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["torch"] = None
        try:
            import torch
        except ImportError:
            print("blocked")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "blocked"
