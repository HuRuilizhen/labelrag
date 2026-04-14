"""Shared test support helpers."""

from __future__ import annotations

from collections.abc import Sequence


class StubEmbeddingProvider:
    """Deterministic embedding provider for test-only semantic ranking."""

    def __init__(self, document_vectors: dict[str, list[float]] | None = None) -> None:
        self._document_vectors = document_vectors or {}

    @property
    def provider_name(self) -> str:
        """Return the provider family name."""

        return "stub"

    @property
    def model_name(self) -> str:
        """Return the provider model name."""

        return "stub-embedding-model"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return deterministic vectors for paragraph texts."""

        return [self._vector_for_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic query vector."""

        return self._vector_for_text(text)

    def _vector_for_text(self, text: str) -> list[float]:
        if text in self._document_vectors:
            return list(self._document_vectors[text])

        lowered = text.lower()
        return [
            1.0 if "language model" in lowered or "language models" in lowered else 0.0,
            1.0 if "developer" in lowered or "developers" in lowered else 0.0,
            1.0 if "production" in lowered or "system" in lowered or "systems" in lowered else 0.0,
        ]
