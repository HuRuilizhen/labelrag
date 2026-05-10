"""Selection helpers for greedy retrieval ranking."""

from collections.abc import Callable

from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.retrieval.coverage import uncovered_overlap_size
from labelrag.types import QueryAnalysis, RetrievedParagraph


def rank_retrieved_paragraphs(
    paragraphs: list[RetrievedParagraph],
) -> list[RetrievedParagraph]:
    """Return a stable descending retrieval order for traced paragraphs."""

    return sorted(
        paragraphs,
        key=lambda item: (
            -item.marginal_gain,
            -(item.semantic_similarity or 0.0),
            -len(item.matched_concept_ids),
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )


def select_greedy_paragraphs(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
    semantic_similarity_for_paragraph: Callable[[str], float],
) -> tuple[list[RetrievedParagraph], bool]:
    """Select paragraphs by greedy coverage over query label IDs.

    Returns the selected paragraphs plus a flag indicating whether semantic
    backfill was needed after greedy coverage completed.
    """

    remaining_label_ids = set(query_analysis.label_ids)
    query_label_ids = set(query_analysis.label_ids)
    query_concept_ids = set(query_analysis.concept_ids)
    label_overlap_candidates: list[RetrievedParagraph] = []

    for paragraph in corpus_index.paragraphs_by_id.values():
        paragraph_label_ids = set(paragraph.label_ids)
        matched_label_ids = sorted(paragraph_label_ids & query_label_ids)
        if not matched_label_ids:
            continue

        matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
        semantic_similarity = semantic_similarity_for_paragraph(paragraph.paragraph_id)
        label_overlap_candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=matched_label_ids,
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=len(matched_concept_ids),
                marginal_gain=0,
                semantic_similarity=semantic_similarity,
                retrieval_score=0.0,
            )
        )

    selected_paragraph_ids: set[str] = set()
    selected: list[RetrievedParagraph] = []
    while remaining_label_ids and len(selected) < max_paragraphs:
        best_candidate: RetrievedParagraph | None = None
        best_sort_key: tuple[int, float, int, int, str] | None = None

        for candidate in label_overlap_candidates:
            if candidate.paragraph_id in selected_paragraph_ids:
                continue

            paragraph_label_ids = set(candidate.paragraph_label_ids)
            gain = uncovered_overlap_size(paragraph_label_ids, remaining_label_ids)
            if gain == 0:
                continue

            newly_covered_label_ids = sorted(paragraph_label_ids & remaining_label_ids)
            already_covered_label_ids = sorted(
                (paragraph_label_ids & query_label_ids) - set(newly_covered_label_ids)
            )
            greedy_candidate = RetrievedParagraph(
                paragraph_id=candidate.paragraph_id,
                text=candidate.text,
                metadata=candidate.metadata,
                newly_covered_label_ids=newly_covered_label_ids,
                already_covered_label_ids=already_covered_label_ids,
                matched_label_ids=list(candidate.matched_label_ids),
                matched_concept_ids=list(candidate.matched_concept_ids),
                paragraph_label_ids=list(candidate.paragraph_label_ids),
                paragraph_concept_ids=list(candidate.paragraph_concept_ids),
                concept_overlap_count=candidate.concept_overlap_count,
                marginal_gain=gain,
                semantic_similarity=candidate.semantic_similarity,
                retrieval_score=float(gain),
            )
            sort_key = (
                gain,
                greedy_candidate.semantic_similarity or 0.0,
                greedy_candidate.concept_overlap_count,
                len(greedy_candidate.paragraph_label_ids),
                _reverse_lexicographic_key(greedy_candidate.paragraph_id),
            )

            if best_candidate is None or best_sort_key is None or sort_key > best_sort_key:
                best_candidate = greedy_candidate
                best_sort_key = sort_key

        if best_candidate is None:
            break

        selected.append(best_candidate)
        selected_paragraph_ids.add(best_candidate.paragraph_id)
        remaining_label_ids -= set(best_candidate.newly_covered_label_ids)

    semantic_backfill_used = False
    if len(selected) < max_paragraphs:
        remaining_candidates = [
            candidate
            for candidate in label_overlap_candidates
            if candidate.paragraph_id not in selected_paragraph_ids
        ]
        if remaining_candidates:
            semantic_backfill_used = True
            ranked_backfill = sorted(
                remaining_candidates,
                key=lambda item: (
                    -(item.semantic_similarity or 0.0),
                    -item.concept_overlap_count,
                    -len(item.paragraph_label_ids),
                    item.paragraph_id,
                ),
            )
            for candidate in ranked_backfill[: max_paragraphs - len(selected)]:
                selected.append(
                    RetrievedParagraph(
                        paragraph_id=candidate.paragraph_id,
                        text=candidate.text,
                        metadata=candidate.metadata,
                        newly_covered_label_ids=[],
                        already_covered_label_ids=list(candidate.matched_label_ids),
                        matched_label_ids=list(candidate.matched_label_ids),
                        matched_concept_ids=list(candidate.matched_concept_ids),
                        paragraph_label_ids=list(candidate.paragraph_label_ids),
                        paragraph_concept_ids=list(candidate.paragraph_concept_ids),
                        concept_overlap_count=candidate.concept_overlap_count,
                        marginal_gain=0,
                        semantic_similarity=candidate.semantic_similarity,
                        retrieval_score=float(candidate.semantic_similarity or 0.0),
                    )
                )

    return selected, semantic_backfill_used


def select_label_gate_semantic_paragraphs(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
    semantic_similarity_for_paragraph: Callable[[str], float],
) -> list[RetrievedParagraph]:
    """Select label-gated candidates and rank them primarily by semantic similarity."""

    query_label_ids = set(query_analysis.label_ids)
    query_concept_ids = set(query_analysis.concept_ids)
    candidates: list[RetrievedParagraph] = []
    for paragraph in corpus_index.paragraphs_by_id.values():
        paragraph_label_ids = set(paragraph.label_ids)
        matched_label_ids = sorted(paragraph_label_ids & query_label_ids)
        if not matched_label_ids:
            continue
        matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
        semantic_similarity = semantic_similarity_for_paragraph(paragraph.paragraph_id)
        candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=list(matched_label_ids),
                already_covered_label_ids=[],
                matched_label_ids=matched_label_ids,
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=len(matched_concept_ids),
                marginal_gain=len(matched_label_ids),
                semantic_similarity=semantic_similarity,
                retrieval_score=float(semantic_similarity),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.semantic_similarity or 0.0),
            -item.marginal_gain,
            -item.concept_overlap_count,
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )
    return ranked[:max_paragraphs]


def select_concept_overlap_fallback(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
) -> list[RetrievedParagraph]:
    """Select paragraphs deterministically by concept overlap for label-free fallback."""

    query_concept_ids = set(query_analysis.concept_ids)
    candidates: list[RetrievedParagraph] = []
    for paragraph in corpus_index.paragraphs_by_id.values():
        matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
        if not matched_concept_ids:
            continue
        candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=len(matched_concept_ids),
                marginal_gain=0,
                semantic_similarity=None,
                retrieval_score=float(len(matched_concept_ids)),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.concept_overlap_count,
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )
    return ranked[:max_paragraphs]


def select_concept_overlap_semantic_fallback(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
    semantic_similarity_for_paragraph: Callable[[str], float],
) -> list[RetrievedParagraph]:
    """Select concept-overlap fallback candidates with semantic reranking."""

    query_concept_ids = set(query_analysis.concept_ids)
    candidates: list[RetrievedParagraph] = []
    for paragraph in corpus_index.paragraphs_by_id.values():
        matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
        if not matched_concept_ids:
            continue
        semantic_similarity = semantic_similarity_for_paragraph(paragraph.paragraph_id)
        candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=len(matched_concept_ids),
                marginal_gain=0,
                semantic_similarity=semantic_similarity,
                retrieval_score=float(len(matched_concept_ids)),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.concept_overlap_count,
            -(item.semantic_similarity or 0.0),
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )
    return ranked[:max_paragraphs]


def select_concept_gate_semantic_fallback(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
    semantic_similarity_for_paragraph: Callable[[str], float],
) -> list[RetrievedParagraph]:
    """Select concept-gated candidates and rank them primarily by semantic similarity."""

    query_concept_ids = set(query_analysis.concept_ids)
    candidates: list[RetrievedParagraph] = []
    for paragraph in corpus_index.paragraphs_by_id.values():
        matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
        if not matched_concept_ids:
            continue
        semantic_similarity = semantic_similarity_for_paragraph(paragraph.paragraph_id)
        candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=len(matched_concept_ids),
                marginal_gain=0,
                semantic_similarity=semantic_similarity,
                retrieval_score=float(semantic_similarity),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.semantic_similarity or 0.0),
            -item.concept_overlap_count,
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )
    return ranked[:max_paragraphs]


def select_semantic_only_fallback(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
    semantic_similarity_for_paragraph: Callable[[str], float],
) -> list[RetrievedParagraph]:
    """Select top-k paragraphs directly by semantic similarity for fallback."""

    del query_analysis
    candidates: list[RetrievedParagraph] = []
    for paragraph in corpus_index.paragraphs_by_id.values():
        semantic_similarity = semantic_similarity_for_paragraph(paragraph.paragraph_id)
        candidates.append(
            RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=[],
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                concept_overlap_count=0,
                marginal_gain=0,
                semantic_similarity=semantic_similarity,
                retrieval_score=float(semantic_similarity),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -(item.semantic_similarity or 0.0),
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )
    return ranked[:max_paragraphs]


def _reverse_lexicographic_key(value: str) -> str:
    """Invert lexicographic order so smaller IDs sort earlier in max comparisons."""

    return "".join(chr(0x10FFFF - ord(character)) for character in value)
