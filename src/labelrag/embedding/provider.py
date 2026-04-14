"""Public embedding provider protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Public interface for paragraph and query embedding providers."""

    @property
    def provider_name(self) -> str:
        """Return the provider family name."""
        ...

    @property
    def model_name(self) -> str:
        """Return the provider model name."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple paragraph texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text."""
        ...
