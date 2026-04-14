"""Small deterministic embedding provider for runnable examples."""

from __future__ import annotations

from collections.abc import Sequence

from labelrag.embedding.provider import EmbeddingProvider


class DemoEmbeddingProvider(EmbeddingProvider):
    """Keyword-based embedding provider that keeps examples runnable offline."""

    @property
    def provider_name(self) -> str:
        """Return the provider name exposed in retrieval metadata."""

        return "demo"

    @property
    def model_name(self) -> str:
        """Return the model name exposed in retrieval metadata."""

        return "demo-keyword-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed paragraph texts deterministically."""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query deterministically."""

        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Map a text to a tiny semantic vector for example purposes."""

        lowered = text.lower()
        return [
            1.0 if "language model" in lowered or "language models" in lowered else 0.0,
            1.0 if "developer" in lowered or "developers" in lowered else 0.0,
            1.0 if "production" in lowered or "system" in lowered else 0.0,
            1.0 if "monitoring" in lowered or "evaluation" in lowered else 0.0,
        ]
