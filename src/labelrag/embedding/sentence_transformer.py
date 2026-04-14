"""Sentence-transformers embedding provider."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, cast

from labelrag.config import EmbeddingConfig


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider backed by `sentence-transformers`."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: object | None = None

    @property
    def provider_name(self) -> str:
        """Return the provider family name."""

        return "sentence-transformers"

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""

        return self._config.model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple paragraph texts."""

        if not texts:
            return []
        model = self._load_model()
        encoded = model.encode(
            list(texts),
            normalize_embeddings=self._config.normalize,
            convert_to_numpy=True,
        )
        return cast(list[list[float]], encoded.tolist())

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text."""

        model = self._load_model()
        encoded = model.encode(
            text,
            normalize_embeddings=self._config.normalize,
            convert_to_numpy=True,
        )
        return cast(list[float], encoded.tolist())

    def _load_model(self) -> Any:
        """Load the configured sentence-transformers model lazily."""

        if self._model is not None:
            return self._model

        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbeddingProvider. "
                "Install labelrag with the embedding dependency path available, for example "
                "`pip install -e .` in this repository."
            ) from error

        sentence_transformer = getattr(sentence_transformers, "SentenceTransformer", None)
        if sentence_transformer is None:
            raise RuntimeError(
                "sentence-transformers.SentenceTransformer is unavailable in the installed package."
            )
        try:
            self._model = sentence_transformer(self._config.model)
        except Exception as error:  # pragma: no cover - exercised via tests with monkeypatch
            raise RuntimeError(
                "Failed to load sentence-transformers model "
                f"{self._config.model!r}. Verify the model name and ensure the model is "
                "available locally or downloadable from Hugging Face."
            ) from error
        return self._model
