"""Ports: `typing.Protocol` interfaces for every replaceable component (plan C.4).

Rules:
- Every type in a port signature is declared in `qf.contracts` or in the lower layer
  `qf.common` (e.g. `ArtifactRef`), never in the packages that implement or consume the port.
- A port is added here in the step that ships its first implementation.
- Ports are `@runtime_checkable`; implementations are registered by name in a `Registry`
  whose `port` attribute points back to the Protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from random import Random
from typing import Protocol, runtime_checkable

from qf.common import ArtifactRef
from qf.contracts.evaluation import CaseScore, MetricValue
from qf.contracts.facts import LoadFacts
from qf.contracts.facts_v2 import AnyLoadFacts
from qf.contracts.generation import GenerationRequest, GenerationResult
from qf.contracts.records import Language, SFTRecord
from qf.contracts.rendering import RenderedRequest, RequestDraft

__all__ = [
    "FactsBuilder",
    "GenerationBackend",
    "HardCase",
    "Metric",
    "RawSource",
    "Splitter",
    "TemplateFamily",
]


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


@runtime_checkable
class TemplateFamily(Protocol):
    """A style of request text in one language (step 6). `ood_only` families are rendered only
    into test_ood/bench."""

    @property
    def name(self) -> str: ...

    @property
    def language(self) -> Language: ...

    @property
    def ood_only(self) -> bool: ...

    def render(self, facts: AnyLoadFacts, draft: RequestDraft, rng: Random) -> RenderedRequest:
        """Text of the draft for `facts`, its evidence and the gold answer built from it."""
        ...


@runtime_checkable
class HardCase(Protocol):
    """A controlled difficulty applied to a draft before rendering (step 6)."""

    @property
    def name(self) -> str: ...

    def applicable(self, facts: AnyLoadFacts) -> bool: ...

    def apply(self, draft: RequestDraft, facts: AnyLoadFacts, rng: Random) -> RequestDraft:
        """A new draft with the difficulty added; the input draft is not changed."""
        ...


@runtime_checkable
class Splitter(Protocol):
    """Assigns splits to records that come without one (step 7); every record of a group gets
    the same split. Parameters (seed, ratios) are the implementation's `Config`."""

    def assign(self, records: Sequence[SFTRecord]) -> list[SFTRecord]:
        """The same records, in the same order, with `split` set."""
        ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Anything that answers chat messages: a fake, LM Studio, llama-server, HF (step 9+)."""

    @property
    def name(self) -> str: ...

    def model_id(self) -> str:
        """The model actually answering (resolved from the backend, never from a file name)."""
        ...

    def generate(self, req: GenerationRequest) -> GenerationResult:
        """One answer. Timeouts and HTTP errors go to `result.error`; nothing is raised."""
        ...


@runtime_checkable
class Metric(Protocol):
    """One aggregated metric (step 9): a value per case and its own aggregation."""

    @property
    def name(self) -> str: ...

    @property
    def higher_is_better(self) -> bool: ...

    def value(self, case: CaseScore) -> MetricValue: ...

    def aggregate(self, values: Sequence[MetricValue]) -> float | None:
        """The metric over cases; None if no case applies (also for an empty list)."""
        ...
