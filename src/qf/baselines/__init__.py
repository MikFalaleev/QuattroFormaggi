"""Lower bounds for the article (D-109): baselines that answer without a language model, run by
the eval harness like any backend. Importing the package registers them in `BACKENDS`."""

from qf.baselines.answers import answer_text, empty_card, request_date
from qf.baselines.backends import EmptyBaseline, RulesBaseline
from qf.baselines.rules import RULES_VERSION, extract_card

__all__ = ["RULES_VERSION", "EmptyBaseline", "RulesBaseline", "answer_text", "empty_card",
           "extract_card", "request_date"]  # fmt: skip
