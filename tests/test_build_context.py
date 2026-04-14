"""Tests for end-to-end retrieval context construction."""

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.types import QueryAnalysis, RetrievedParagraph
from support import StubEmbeddingProvider


def test_build_context_requires_fit() -> None:
    """Context building should fail clearly before the pipeline is fitted."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())

    with pytest.raises(RuntimeError, match="requires fit"):
        pipeline.build_context("How do developers use language models?")


def test_build_context_returns_prompt_and_metadata() -> None:
    """Context building should return retrieval results, prompt text, and trace metadata."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    result = pipeline.build_context("How do developers use language models?")

    assert result.question == "How do developers use language models?"
    assert result.query_analysis.query_text == result.question
    assert result.metadata["retrieval_strategy"] == "greedy_label_coverage_semantic_rerank"
    assert result.metadata["embedding_provider"] == "stub"
    assert result.metadata["embedding_model"] == "stub-embedding-model"
    assert result.metadata["semantic_reranking_enabled"] is True
    assert (
        result.metadata["label_free_fallback_strategy"]
        == "concept_overlap_semantic_rerank"
    )
    assert result.metadata["query_label_ids"] == result.query_analysis.label_ids
    assert result.metadata["retrieval_limit"] == pipeline.config.retrieval.max_paragraphs
    assert result.metadata["used_label_free_fallback"] is False
    assert result.metadata["full_label_coverage_met"] is True
    assert result.metadata["attempted_covered_label_ids"] == result.metadata["covered_label_ids"]
    assert (
        result.metadata["attempted_uncovered_label_ids"]
        == result.metadata["uncovered_label_ids"]
    )
    assert "covered_label_ids" in result.metadata
    assert "uncovered_label_ids" in result.metadata
    assert result.prompt_context
    assert "[Paragraph 1" in result.prompt_context


def test_build_context_respects_prompt_configuration() -> None:
    """Context building should honor the prompt rendering configuration."""

    config = RAGPipelineConfig()
    config.prompt.include_paragraph_ids = False
    config.prompt.max_context_characters = 20
    pipeline = RAGPipeline(config, embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("How do developers use language models?")

    assert result.prompt_context.startswith("[Paragraph 1]")
    assert "| id=" not in result.prompt_context
    assert len(result.prompt_context) <= 20


def test_build_context_uses_default_label_free_fallback_strategy() -> None:
    """Label-free queries should use the default configured fallback strategy."""

    pipeline = RAGPipeline(
        RAGPipelineConfig(),
        embedding_provider=StubEmbeddingProvider(
            {"Quantum batteries improve starship reactors.": [0.25, 0.25, 0.25]}
        ),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("Quantum batteries improve starship reactors.")

    assert result.metadata["used_label_free_fallback"] is True
    assert result.metadata["retrieval_strategy"] == "concept_overlap_semantic_fallback"
    assert (
        result.metadata["label_free_fallback_strategy"]
        == "concept_overlap_semantic_rerank"
    )
    assert result.metadata["covered_label_ids"] == []
    assert result.metadata["uncovered_label_ids"] == []
    assert result.metadata["attempted_covered_label_ids"] == []
    assert result.metadata["attempted_uncovered_label_ids"] == []
    assert result.metadata["semantic_reranking_enabled"] is False


def test_build_context_supports_concept_overlap_only_fallback() -> None:
    """Fallback strategy should support pure concept-overlap ordering."""

    config = RAGPipelineConfig()
    config.retrieval.label_free_fallback_strategy = "concept_overlap_only"
    pipeline = RAGPipeline(
        config,
        embedding_provider=StubEmbeddingProvider(
            {"Quantum batteries improve starship reactors.": [0.25, 0.25, 0.25]}
        ),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("Quantum batteries improve starship reactors.")

    assert result.metadata["used_label_free_fallback"] is True
    assert result.metadata["retrieval_strategy"] == "concept_overlap_only_fallback"
    assert result.metadata["semantic_reranking_enabled"] is False


def test_build_context_supports_semantic_only_fallback() -> None:
    """Fallback strategy should support full semantic-only ranking."""

    config = RAGPipelineConfig()
    config.retrieval.label_free_fallback_strategy = "semantic_only"
    pipeline = RAGPipeline(
        config,
        embedding_provider=StubEmbeddingProvider(
            {"Quantum batteries improve starship reactors.": [0.25, 0.25, 0.25]}
        ),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("Quantum batteries improve starship reactors.")

    assert result.metadata["used_label_free_fallback"] is True
    assert result.metadata["retrieval_strategy"] == "semantic_only_fallback"
    assert result.metadata["semantic_reranking_enabled"] is True
    assert len(result.retrieved_paragraphs) == 2


def test_build_context_short_circuits_semantic_overlap_fallback_without_concepts() -> None:
    """Concept-overlap semantic fallback should return early when the query has no concepts."""

    pipeline = RAGPipeline(
        RAGPipelineConfig(),
        embedding_provider=StubEmbeddingProvider(),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("???")

    assert result.retrieved_paragraphs == []
    assert result.metadata["used_label_free_fallback"] is True
    assert result.metadata["retrieval_strategy"] == "concept_overlap_semantic_fallback"
    assert result.metadata["semantic_reranking_enabled"] is False


def test_build_context_can_disable_label_free_fallback() -> None:
    """Label-free fallback should be configurable."""

    config = RAGPipelineConfig()
    config.retrieval.allow_label_free_fallback = False
    pipeline = RAGPipeline(config, embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("Quantum batteries improve starship reactors.")

    assert result.retrieved_paragraphs == []
    assert result.prompt_context == ""
    assert result.metadata["used_label_free_fallback"] is False
    assert result.metadata["retrieval_strategy"] == "no_retrieval"
    assert result.metadata["attempted_covered_label_ids"] == []
    assert result.metadata["attempted_uncovered_label_ids"] == []
    assert result.metadata["semantic_reranking_enabled"] is False


def test_build_context_can_require_full_label_coverage() -> None:
    """Requiring full label coverage should suppress partial retrieval results."""

    class StubPipeline(RAGPipeline):
        def __init__(self, config: RAGPipelineConfig) -> None:
            super().__init__(config)
            self._corpus_index = CorpusIndex()

        def analyze_query(self, question: str) -> QueryAnalysis:
            del question
            return QueryAnalysis(
                query_text="stub question",
                concepts=["developers", "monitoring"],
                concept_ids=["c1", "c2"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "monitoring"],
            )

        def _retrieve_paragraphs(
            self,
            query_analysis: QueryAnalysis,
        ) -> tuple[list[RetrievedParagraph], bool, str]:
            del query_analysis
            return (
                [
                    RetrievedParagraph(
                        paragraph_id="p1",
                        text="Paragraph 1",
                        metadata=None,
                        newly_covered_label_ids=["l1"],
                        already_covered_label_ids=[],
                        matched_label_ids=["l1"],
                        matched_concept_ids=["c1"],
                        paragraph_label_ids=["l1"],
                        paragraph_concept_ids=["c1"],
                        concept_overlap_count=1,
                        marginal_gain=1,
                        semantic_similarity=0.25,
                        retrieval_score=1.0,
                    )
                ],
                True,
                "greedy_label_coverage_semantic_rerank",
            )

    config = RAGPipelineConfig()
    config.retrieval.require_full_label_coverage = True
    pipeline = StubPipeline(config)

    result = pipeline.build_context("How do developers use language models and monitoring?")

    assert result.retrieved_paragraphs == []
    assert result.prompt_context == ""
    assert result.metadata["require_full_label_coverage"] is True
    assert result.metadata["full_label_coverage_met"] is False
    assert result.metadata["covered_label_ids"] == []
    assert result.metadata["uncovered_label_ids"] == ["l1", "l2"]
    assert result.metadata["attempted_covered_label_ids"] == ["l1"]
    assert result.metadata["attempted_uncovered_label_ids"] == ["l2"]
