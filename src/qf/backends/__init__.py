"""Generation backends: adapters of the `GenerationBackend` port (plan C.2).

Backends get ready messages from the caller and never build prompts (they do not use
`qf.domain`). Light backends register here; heavy ones (hf_local) lazily in `cli.wiring`.
"""

from qf.backends.fake import TIMEOUT, FakeBackend
from qf.backends.openai_local import LocalOpenAIClient, OpenAILocalBackend
from qf.backends.registry import BACKENDS

__all__ = ["BACKENDS", "TIMEOUT", "FakeBackend", "LocalOpenAIClient", "OpenAILocalBackend"]
