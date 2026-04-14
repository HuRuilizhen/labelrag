"""Tests for query analysis against the fitted label space."""

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig
from support import StubEmbeddingProvider


def test_analyze_query_requires_fit() -> None:
    """Query analysis should fail clearly when the pipeline is not fitted."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())

    with pytest.raises(RuntimeError, match="requires fit"):
        pipeline.analyze_query("What do developers use?")


def test_analyze_query_returns_structured_query_analysis() -> None:
    """Query analysis should reuse the fitted label space and expose concepts and labels."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    analysis = pipeline.analyze_query("How do developers use language models?")

    assert analysis.query_text == "How do developers use language models?"
    assert analysis.concept_ids
    assert analysis.concepts
    assert len(analysis.concepts) == len(analysis.concept_ids)
    assert len(analysis.label_display_names) == len(analysis.label_ids)


def test_analyze_query_uses_fitted_corpus_label_space() -> None:
    """Query analysis should not introduce concepts that are absent from the fitted corpus."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    analysis = pipeline.analyze_query("Quantum batteries improve starship reactors.")

    assert "quantum batteries" not in analysis.concepts
    assert "starship reactors" not in analysis.concepts
