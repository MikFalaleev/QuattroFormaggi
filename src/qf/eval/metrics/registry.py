"""Registry of metrics (plan C.5). A run computes the metrics named in its config."""

from __future__ import annotations

from qf.common import Registry
from qf.contracts import Metric

__all__ = ["METRICS"]

METRICS: Registry[type[Metric]] = Registry("metric", port=Metric)
