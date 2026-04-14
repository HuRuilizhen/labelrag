"""Shared test fixtures and stubs."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig
from support import StubEmbeddingProvider


@pytest.fixture
def embedding_provider() -> StubEmbeddingProvider:
    """Return the default deterministic embedding provider."""

    return StubEmbeddingProvider()


@pytest.fixture
def pipeline_factory() -> Callable[..., RAGPipeline]:
    """Return a factory that builds pipelines with a stub embedding provider."""

    def _factory(
        config: RAGPipelineConfig | None = None,
        *,
        embedding_provider: StubEmbeddingProvider | None = None,
    ) -> RAGPipeline:
        return RAGPipeline(
            config or RAGPipelineConfig(),
            embedding_provider=embedding_provider or StubEmbeddingProvider(),
        )

    return _factory
