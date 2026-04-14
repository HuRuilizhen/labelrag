"""Tests for greedy label-based and fallback retrieval."""

from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.retrieval.selector import (
    select_concept_overlap_fallback,
    select_concept_overlap_semantic_fallback,
    select_greedy_paragraphs,
    select_semantic_only_fallback,
)
from labelrag.types import IndexedParagraph, QueryAnalysis


def test_select_greedy_paragraphs_covers_query_labels_greedily() -> None:
    """The selector should choose paragraphs that greedily cover remaining labels."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c2"],
                concept_texts=["systems"],
                label_ids=["l2"],
                label_display_names=["systems"],
            ),
            "p3": IndexedParagraph(
                paragraph_id="p3",
                text="Paragraph 3",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "systems"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=3,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.2,
            "p2": 0.3,
            "p3": 0.9,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p3"]
    assert selected[0].marginal_gain == 2
    assert selected[0].semantic_similarity == 0.9
    assert selected[0].newly_covered_label_ids == ["l1", "l2"]
    assert selected[0].already_covered_label_ids == []


def test_select_greedy_paragraphs_uses_semantic_similarity_as_tiebreak() -> None:
    """Semantic similarity should break ties before concept overlap."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "language models"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use language models?",
        concepts=["developers", "language models"],
        concept_ids=["c1", "c2"],
        label_ids=["l1"],
        label_display_names=["developers"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.1,
            "p2": 0.8,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p2"]


def test_select_greedy_paragraphs_uses_paragraph_id_as_final_tiebreak() -> None:
    """Lexicographically smaller paragraph IDs should win final ties."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers work?",
        concepts=["developers"],
        concept_ids=["c1"],
        label_ids=["l1"],
        label_display_names=["developers"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
        semantic_similarity_for_paragraph=lambda _paragraph_id: 0.5,
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p1"]


def test_select_greedy_paragraphs_respects_max_paragraphs() -> None:
    """Selection should stop when the configured paragraph limit is reached."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c2"],
                concept_texts=["systems"],
                label_ids=["l2"],
                label_display_names=["systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=1,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.2,
            "p2": 0.8,
        }[paragraph_id],
    )

    assert len(selected) == 1


def test_select_greedy_paragraphs_tracks_already_covered_labels() -> None:
    """Later retrieval steps should distinguish newly covered from already covered labels."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "systems"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.2,
            "p2": 0.9,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p2"]


def test_select_greedy_paragraphs_prefers_gain_over_semantic_similarity() -> None:
    """Marginal gain should remain the primary greedy objective."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "language models"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.1,
            "p2": 0.9,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p1"]


def test_select_concept_overlap_fallback_preserves_overlap_ordering() -> None:
    """Pure concept-overlap fallback should ignore semantic similarity."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "systems"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=[],
        label_display_names=[],
    )

    selected = select_concept_overlap_fallback(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p1", "p2"]
    assert selected[0].semantic_similarity is None


def test_select_concept_overlap_semantic_fallback_uses_semantic_tiebreak() -> None:
    """Concept-overlap semantic fallback should use similarity within the overlap candidate set."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers work?",
        concepts=["developers"],
        concept_ids=["c1"],
        label_ids=[],
        label_display_names=[],
    )

    selected = select_concept_overlap_semantic_fallback(
        query_analysis,
        corpus_index,
        max_paragraphs=2,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.2,
            "p2": 0.8,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p2", "p1"]
    assert selected[0].semantic_similarity == 0.8


def test_select_semantic_only_fallback_uses_global_similarity_top_k() -> None:
    """Semantic-only fallback should rank over the full paragraph set."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=[],
                concept_texts=[],
                label_ids=["l2"],
                label_display_names=["systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="Quantum batteries",
        concepts=[],
        concept_ids=[],
        label_ids=[],
        label_display_names=[],
    )

    selected = select_semantic_only_fallback(
        query_analysis,
        corpus_index,
        max_paragraphs=1,
        semantic_similarity_for_paragraph=lambda paragraph_id: {
            "p1": 0.1,
            "p2": 0.9,
        }[paragraph_id],
    )

    assert [paragraph.paragraph_id for paragraph in selected] == ["p2"]
    assert selected[0].concept_overlap_count == 0
    assert selected[0].semantic_similarity == 0.9
