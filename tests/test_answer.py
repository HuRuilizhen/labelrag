"""Tests for answer generation flows."""

from dataclasses import dataclass

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.generation.generator import GeneratedAnswer
from support import StubEmbeddingProvider


@dataclass
class StubGenerator:
    """Simple synchronous generator stub for tests."""

    answer_text: str
    model: str = "stub-model"

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Return a deterministic generated answer payload."""

        return GeneratedAnswer(
            text=f"{self.answer_text} | q={question} | ctx={bool(context)}",
            metadata={"model": self.model},
        )


def test_answer_returns_empty_text_when_no_generator_is_configured() -> None:
    """answer should still return a structured result without a generator."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.answer("How do developers use language models?")

    assert result.answer_text == ""
    assert result.metadata["generator_name"] == ""
    assert result.metadata["generation_model"] == ""
    assert result.metadata["generation_metadata"] == {}


def test_answer_uses_pipeline_level_generator_when_configured() -> None:
    """answer should use the generator configured on the pipeline."""

    pipeline = RAGPipeline(
        RAGPipelineConfig(),
        generator=StubGenerator("pipeline"),
        embedding_provider=StubEmbeddingProvider(),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.answer("How do developers use language models?")

    assert "pipeline" in result.answer_text
    assert result.metadata["generator_name"] == "StubGenerator"
    assert result.metadata["generation_model"] == "stub-model"


def test_answer_with_generator_overrides_pipeline_generator() -> None:
    """answer_with_generator should use the per-call generator override."""

    pipeline = RAGPipeline(
        RAGPipelineConfig(),
        generator=StubGenerator("pipeline"),
        embedding_provider=StubEmbeddingProvider(),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.answer_with_generator(
        "How do developers use language models?",
        StubGenerator("override", model="override-model"),
    )

    assert "override" in result.answer_text
    assert result.metadata["generator_name"] == "StubGenerator"
    assert result.metadata["generation_model"] == "override-model"


def test_answer_metadata_preserves_retrieval_trace_fields() -> None:
    """answer results should keep the retrieval metadata contract."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.answer("How do developers use language models?")

    assert "covered_label_ids" in result.metadata
    assert "uncovered_label_ids" in result.metadata
    assert result.metadata["retrieval_strategy"] == "greedy_label_coverage_semantic_rerank"
