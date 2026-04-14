"""Tests for corpus fitting and index construction."""

from typing import TypedDict

import pytest
from labelgen import Paragraph

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.pipeline import rag_pipeline as rag_pipeline_module
from support import FailingEmbeddingProvider, StubEmbeddingProvider


class SerializedParagraphPayload(TypedDict):
    """Serialized paragraph payload reconstructed by callers before fit."""

    id: str
    text: str
    metadata: dict[str, str]


def test_fit_returns_pipeline_for_string_inputs() -> None:
    """Fitting on raw paragraph strings should return the pipeline itself."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())

    fitted = pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    assert fitted is pipeline
    assert pipeline.fit_result is not None
    assert pipeline.corpus_index is not None
    assert len(pipeline.corpus_index.paragraphs_by_id) == 3


def test_fit_accepts_paragraph_objects_and_preserves_metadata() -> None:
    """Fitting should accept labelgen Paragraph objects as input."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())

    pipeline.fit(
        [
            Paragraph(
                id="",
                text="OpenAI builds language models for developers.",
                metadata={"doc_id": "doc-1", "title": "Doc One"},
            ),
            Paragraph(
                id="",
                text="Developers use language models in production systems.",
                metadata={"doc_id": "doc-1", "title": "Doc One"},
            ),
        ]
    )

    assert pipeline.corpus_index is not None
    assert "doc-1#p0" in pipeline.corpus_index.paragraphs_by_id
    assert (
        pipeline.corpus_index.paragraphs_by_id["doc-1#p0"].metadata
        == {"doc_id": "doc-1", "title": "Doc One"}
    )


def test_fit_accepts_reconstructed_serialized_paragraph_payloads() -> None:
    """Fitting should accept Paragraph objects reconstructed from serialized payloads."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    serialized_payloads: list[SerializedParagraphPayload] = [
        {
            "id": "",
            "text": "OpenAI builds language models for developers.",
            "metadata": {"doc_id": "doc-2", "title": "Serialized Doc"},
        },
        {
            "id": "",
            "text": "Developers use language models in production systems.",
            "metadata": {"doc_id": "doc-2", "title": "Serialized Doc"},
        },
    ]
    reconstructed_paragraphs = [Paragraph(**payload) for payload in serialized_payloads]

    pipeline.fit(reconstructed_paragraphs)

    assert pipeline.corpus_index is not None
    assert "doc-2#p0" in pipeline.corpus_index.paragraphs_by_id
    assert (
        pipeline.corpus_index.paragraphs_by_id["doc-2#p0"].metadata
        == {"doc_id": "doc-2", "title": "Serialized Doc"}
    )


def test_fit_builds_label_and_concept_lookup_tables() -> None:
    """Fitting should build paragraph, concept, and label lookup tables."""

    pipeline = RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need deployment tooling.",
        ]
    )

    assert pipeline.corpus_index is not None
    assert pipeline.corpus_index.label_display_names_by_id

    paragraph_record = next(iter(pipeline.corpus_index.paragraphs_by_id.values()))
    assert paragraph_record.concept_ids
    assert paragraph_record.concept_texts
    assert pipeline.corpus_index.concept_ids_by_paragraph[paragraph_record.paragraph_id]
    assert pipeline.corpus_index.concept_texts_by_id

    if paragraph_record.label_ids:
        first_label_id = paragraph_record.label_ids[0]
        assert first_label_id in pipeline.corpus_index.label_display_names_by_id
        assert (
            paragraph_record.paragraph_id
            in pipeline.corpus_index.paragraph_ids_by_label[first_label_id]
        )

    first_concept_id = paragraph_record.concept_ids[0]
    assert (
        paragraph_record.paragraph_id
        in pipeline.corpus_index.paragraph_ids_by_concept[first_concept_id]
    )


def test_fit_requires_embedding_provider() -> None:
    """Unsupported configured providers should fail clearly during construction."""

    config = RAGPipelineConfig()
    config.embedding.provider = "unsupported-provider"

    with pytest.raises(RuntimeError, match="Unsupported embedding provider"):
        RAGPipeline(config)


def test_fit_uses_default_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipelines should fit successfully with the config-driven default provider."""

    def build_stub_provider(_: object) -> StubEmbeddingProvider:
        return StubEmbeddingProvider()

    monkeypatch.setattr(
        rag_pipeline_module,
        "SentenceTransformerEmbeddingProvider",
        build_stub_provider,
    )
    pipeline = RAGPipeline(RAGPipelineConfig())

    pipeline.fit(["OpenAI builds language models for developers."])

    assert pipeline.fit_result is not None
    assert pipeline.corpus_index is not None


def test_explicit_embedding_provider_overrides_configured_provider() -> None:
    """Explicit provider injection should override configured provider resolution."""

    config = RAGPipelineConfig()
    config.embedding.provider = "unsupported-provider"
    pipeline = RAGPipeline(config, embedding_provider=StubEmbeddingProvider())

    pipeline.fit(["OpenAI builds language models for developers."])

    assert pipeline.fit_result is not None


def test_fit_does_not_leave_partial_state_after_embedding_failure() -> None:
    """Embedding failures should not leave the pipeline in a half-fitted state."""

    pipeline = RAGPipeline(
        RAGPipelineConfig(),
        embedding_provider=FailingEmbeddingProvider(),
    )

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        pipeline.fit(["OpenAI builds language models for developers."])

    assert pipeline.fit_result is None
    assert pipeline.corpus_index is None

    with pytest.raises(RuntimeError, match="requires fit"):
        pipeline.build_context("How do developers use language models?")
