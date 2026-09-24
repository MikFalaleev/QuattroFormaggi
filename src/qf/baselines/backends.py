"""The lower bounds as generation backends (D-109): the eval harness runs them like any model,
on the same benchmark, prompt and metrics.

- `baseline_empty` answers every request with the empty card (all null, no conditions) and
  the missing fields of the rule: what the metrics give a model that knows nothing but writes
  a perfect answer format.
- `baseline_rules` answers with the card of the rule-based extractor (`rules.py`).
"""

from __future__ import annotations

import time

from qf.backends import BACKENDS
from qf.baselines.answers import answer_text, empty_card, refuse, request_date, user_text
from qf.baselines.rules import RULES_VERSION, extract_card
from qf.common import QFError, StrictConfig
from qf.contracts import CARD_V2_SCHEMA_VERSION, GenerationRequest, GenerationResult
from qf.contracts import TargetSchemaVersion as SchemaVersion

__all__ = ["EmptyBaseline", "RulesBaseline"]


@BACKENDS.register("baseline_empty")
class EmptyBaseline:
    """The empty card with the missing fields of the rule, whatever the request."""

    class Config(StrictConfig):
        schema_version: SchemaVersion = CARD_V2_SCHEMA_VERSION

    name = "baseline_empty"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._answer = answer_text(config.schema_version, empty_card(config.schema_version))

    def model_id(self) -> str:
        return f"baseline:empty@{self.config.schema_version}"

    def generate(self, req: GenerationRequest) -> GenerationResult:
        return refuse(req, self.name) or GenerationResult(text=self._answer, latency_s=0.0)


@BACKENDS.register("baseline_rules")
class RulesBaseline:
    """The card found by regular expressions and keyword lists (card_v2 only)."""

    class Config(StrictConfig):
        schema_version: SchemaVersion = CARD_V2_SCHEMA_VERSION

    name = "baseline_rules"

    def __init__(self, config: Config) -> None:
        if config.schema_version != CARD_V2_SCHEMA_VERSION:
            raise QFError(f"baseline_rules answers card_v2 only, not {config.schema_version}")
        self.config = config

    def model_id(self) -> str:
        return f"baseline:{RULES_VERSION}@{self.config.schema_version}"

    def generate(self, req: GenerationRequest) -> GenerationResult:
        refused = refuse(req, self.name)
        if refused is not None:
            return refused
        started = time.monotonic()
        text = user_text(req)
        card = extract_card(text, request_date(text))
        answer = answer_text(self.config.schema_version, card)
        return GenerationResult(text=answer, latency_s=round(time.monotonic() - started, 6))
