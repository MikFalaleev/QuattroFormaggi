"""Ports: `typing.Protocol` interfaces for every replaceable component (plan C.4).

Rules:
- Every type in a port signature is declared in `qf.contracts` or in the lower layer
  `qf.common` (e.g. `ArtifactRef`), never in the packages that implement or consume the port.
- A port is added here in the step that ships its first implementation.
- Ports are `@runtime_checkable`; implementations are registered by name in a `Registry`
  whose `port` attribute points back to the Protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from qf.common import ArtifactRef
from qf.contracts.facts import LoadFacts

__all__ = ["FactsBuilder", "RawSource"]


@runtime_checkable
class RawSource(Protocol):
    """A raw dataset fetched into `dest_root` as one `raw_dataset` artifact (step 2)."""

    def target(self, dest_root: Path) -> Path:
        """Directory where the artifact is (or will be) stored under `dest_root`."""
        ...

    def fetch(self, dest_root: Path, run_id: str) -> ArtifactRef:
        """Download into `target(dest_root)`, which must not exist yet; write provenance and
        the artifact manifest; return the artifact reference produced by `run_id`."""
        ...

    def verify(self, ref: ArtifactRef) -> list[str]:
        """Discrepancies between the stored artifact, its provenance and the source config.
        An empty list means the artifact is intact."""
        ...


@runtime_checkable
class FactsBuilder(Protocol):
    """Turns one raw dataset artifact into facts about individual shipments (step 5)."""

    def build(self, raw: ArtifactRef, root: Path) -> Iterable[LoadFacts]:
        """Facts from the verified raw artifact `raw`; its path is relative to `root`.
        Problems in the data raise `DataValidationError` listing every bad record."""
        ...

    def inputs(self, root: Path) -> dict[str, str]:
        """Files besides the raw artifact that `build` reads (path relative to `root` ->
        sha256), recorded in the run manifest; empty if there are none."""
        ...
