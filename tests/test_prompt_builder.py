"""Tests for prompt context rendering."""

from labelrag.config import PromptConfig
from labelrag.generation.prompt_builder import build_prompt_context
from labelrag.types import RetrievedParagraph


def test_build_prompt_context_includes_paragraph_ids_by_default() -> None:
    """Default prompt rendering should include stable paragraph identifiers."""

    result = build_prompt_context(
        [
            RetrievedParagraph(
                paragraph_id="p1",
                text="Paragraph text.",
                metadata=None,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=[],
                paragraph_label_ids=[],
                paragraph_concept_ids=[],
                concept_overlap_count=0,
                marginal_gain=1,
                semantic_similarity=0.5,
                retrieval_score=1.0,
            )
        ],
        PromptConfig(),
    )

    assert "[Paragraph 1 | id=p1]" in result
    assert "Paragraph text." in result


def test_build_prompt_context_respects_character_limit() -> None:
    """Prompt rendering should apply the configured hard character cap."""

    result = build_prompt_context(
        [
            RetrievedParagraph(
                paragraph_id="p1",
                text="abcdefghij",
                metadata=None,
                newly_covered_label_ids=[],
                already_covered_label_ids=[],
                matched_label_ids=[],
                matched_concept_ids=[],
                paragraph_label_ids=[],
                paragraph_concept_ids=[],
                concept_overlap_count=0,
                marginal_gain=1,
                semantic_similarity=0.5,
                retrieval_score=1.0,
            )
        ],
        PromptConfig(max_context_characters=8),
    )

    assert result == "[Paragra"


def test_build_prompt_context_can_include_label_annotations() -> None:
    """Prompt rendering should optionally include paragraph label annotations."""

    result = build_prompt_context(
        [
            RetrievedParagraph(
                paragraph_id="p1",
                text="Paragraph text.",
                metadata=None,
                newly_covered_label_ids=["l1"],
                already_covered_label_ids=[],
                matched_label_ids=["l1"],
                matched_concept_ids=[],
                paragraph_label_ids=["l1", "l2"],
                paragraph_concept_ids=[],
                concept_overlap_count=0,
                marginal_gain=1,
                semantic_similarity=0.5,
                retrieval_score=1.0,
            )
        ],
        PromptConfig(include_label_annotations=True),
    )

    assert "Labels: l1, l2" in result
