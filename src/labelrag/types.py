"""Public data models for `labelrag`."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexedParagraph:
    """Paragraph-side index record used for retrieval and inspection."""

    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    concept_ids: list[str]
    concept_texts: list[str]
    label_ids: list[str]
    label_display_names: list[str]


@dataclass(slots=True)
class LabelRecord:
    """Label-side inspection record built from fitted retrieval state."""

    label_id: str
    display_name: str
    concept_ids: list[str]
    paragraph_ids: list[str]


@dataclass(slots=True)
class ConceptRecord:
    """Concept-side inspection record built from fitted retrieval state."""

    concept_id: str
    text: str
    paragraph_ids: list[str]


@dataclass(slots=True)
class QueryAnalysis:
    """Structured query-side analysis derived from a fitted label space."""

    query_text: str
    concepts: list[str]
    concept_ids: list[str]
    label_ids: list[str]
    label_display_names: list[str]


@dataclass(slots=True)
class RetrievedParagraph:
    """One retrieved paragraph plus its coverage trace."""

    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    newly_covered_label_ids: list[str]
    already_covered_label_ids: list[str]
    matched_label_ids: list[str]
    matched_concept_ids: list[str]
    paragraph_label_ids: list[str]
    paragraph_concept_ids: list[str]
    concept_overlap_count: int
    marginal_gain: int
    retrieval_score: float


@dataclass(slots=True)
class RetrievalResult:
    """Retrieval-only output returned by `build_context`."""

    question: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(slots=True)
class RAGAnswerResult:
    """Answer output returned by `answer`."""

    question: str
    answer_text: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any] = field(default_factory=lambda: {})
