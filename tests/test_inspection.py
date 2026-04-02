"""Tests for inspection and lookup APIs."""

from pathlib import Path

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig


def _build_fitted_pipeline() -> RAGPipeline:
    """Construct a fitted pipeline with stable paragraph metadata."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )
    return pipeline


def test_lookup_methods_require_fit() -> None:
    """Inspection methods should require a fitted pipeline."""

    pipeline = RAGPipeline(RAGPipelineConfig())

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_paragraph("p0")

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_label_paragraph_ids("l1")

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_label_paragraphs("l1")

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_paragraph_label_ids("p0")

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_paragraph_concept_ids("p0")

    with pytest.raises(RuntimeError, match="requires fit\\(\\)"):
        pipeline.get_concept_paragraph_ids("c1")


def test_get_paragraph_returns_none_for_unknown_id() -> None:
    """Unknown paragraph IDs should return None."""

    pipeline = _build_fitted_pipeline()

    assert pipeline.get_paragraph("does-not-exist") is None


def test_lookup_methods_return_empty_lists_for_unknown_ids() -> None:
    """Unknown label and concept IDs should return empty lists."""

    pipeline = _build_fitted_pipeline()

    assert pipeline.get_label_paragraph_ids("does-not-exist") == []
    assert pipeline.get_label_paragraphs("does-not-exist") == []
    assert pipeline.get_paragraph_label_ids("does-not-exist") == []
    assert pipeline.get_paragraph_concept_ids("does-not-exist") == []
    assert pipeline.get_concept_paragraph_ids("does-not-exist") == []


def test_lookup_methods_return_stable_index_records() -> None:
    """Lookup methods should expose stable paragraph, label, and concept mappings."""

    pipeline = _build_fitted_pipeline()
    assert pipeline.corpus_index is not None

    paragraph_id = sorted(pipeline.corpus_index.paragraphs_by_id)[0]
    paragraph = pipeline.get_paragraph(paragraph_id)

    assert paragraph is not None
    assert paragraph.paragraph_id == paragraph_id
    assert paragraph.text
    assert pipeline.get_paragraph_label_ids(paragraph_id) == paragraph.label_ids
    assert pipeline.get_paragraph_concept_ids(paragraph_id) == paragraph.concept_ids

    if paragraph.label_ids:
        label_id = paragraph.label_ids[0]
        paragraph_ids = pipeline.get_label_paragraph_ids(label_id)
        paragraphs = pipeline.get_label_paragraphs(label_id)

        assert paragraph_ids
        assert paragraph_id in paragraph_ids
        assert [item.paragraph_id for item in paragraphs] == paragraph_ids

    if paragraph.concept_ids:
        concept_id = paragraph.concept_ids[0]
        assert paragraph_id in pipeline.get_concept_paragraph_ids(concept_id)
        assert concept_id in pipeline.corpus_index.concept_texts_by_id


def test_loaded_pipeline_preserves_inspection_behavior(tmp_path: Path) -> None:
    """Save/load should preserve inspection lookups."""

    pipeline = _build_fitted_pipeline()
    assert pipeline.corpus_index is not None

    paragraph_id = sorted(pipeline.corpus_index.paragraphs_by_id)[0]
    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")

    loaded = RAGPipeline.load(output_dir)
    assert loaded.get_paragraph(paragraph_id) == pipeline.get_paragraph(paragraph_id)
    assert loaded.get_paragraph_label_ids(paragraph_id) == pipeline.get_paragraph_label_ids(
        paragraph_id
    )
    assert loaded.get_paragraph_concept_ids(paragraph_id) == pipeline.get_paragraph_concept_ids(
        paragraph_id
    )

    paragraph = pipeline.get_paragraph(paragraph_id)
    assert paragraph is not None
    if paragraph.concept_ids:
        concept_id = paragraph.concept_ids[0]
        assert loaded.get_concept_paragraph_ids(concept_id) == pipeline.get_concept_paragraph_ids(
            concept_id
        )
