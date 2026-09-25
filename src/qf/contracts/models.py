"""A reference to a pinned model on the Hugging Face Hub (plan step 12)."""

from __future__ import annotations

from pydantic import Field

from qf.contracts._model import ContractModel

__all__ = ["ModelRef"]


class ModelRef(ContractModel):
    id: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # a full commit sha, never a branch
