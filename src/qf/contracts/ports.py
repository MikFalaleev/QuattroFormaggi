"""Ports: `typing.Protocol` interfaces for every replaceable component (plan C.4).

Rules:
- Every type that appears in a port signature is declared in `qf.contracts`, never in the
  packages that implement or consume the port.
- A port is added here in the step that ships its first implementation, together with the
  types it needs. The first port (`RawSource`) arrives in step 2.
- Ports are `@runtime_checkable`; implementations are registered by name in a `Registry`
  whose `port` attribute points back to the Protocol.
"""

from __future__ import annotations

__all__: list[str] = []
