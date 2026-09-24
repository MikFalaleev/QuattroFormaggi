"""Local web interface for trying the pipeline by hand (`qf ui`, D-107).

A composition layer like the CLI: it reads the data, the benchmarks and the eval runs, scores
an answer typed by hand and renders new requests with the generator. No model is involved.
"""

from qf.cli.webui.app import DATASETS, GENERATED, WebApp, WebError
from qf.cli.webui.server import LOOPBACK_HOSTS, MAX_BODY_BYTES, make_server

__all__ = ["DATASETS", "GENERATED", "LOOPBACK_HOSTS", "MAX_BODY_BYTES", "WebApp", "WebError",
           "make_server"]  # fmt: skip
