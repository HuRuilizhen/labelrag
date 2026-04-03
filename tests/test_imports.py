"""Smoke tests for package imports and public exports."""

from labelrag import (
    ConceptRecord,
    LabelRecord,
    Paragraph,
    RAGPipeline,
    RAGPipelineConfig,
    RetrievalConfig,
)


def test_public_imports_are_available() -> None:
    """The package should expose the expected top-level names."""

    assert Paragraph is not None
    assert LabelRecord is not None
    assert ConceptRecord is not None
    assert RAGPipeline is not None
    assert RAGPipelineConfig is not None
    assert RetrievalConfig is not None
