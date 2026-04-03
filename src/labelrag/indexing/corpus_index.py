"""Corpus index models and helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from labelgen import LabelGenerationResult

from labelrag.types import IndexedParagraph


@dataclass(slots=True)
class CorpusIndex:
    """Stored paragraph-side retrieval artifacts."""

    paragraphs_by_id: dict[str, IndexedParagraph] = field(default_factory=lambda: {})
    paragraph_ids_by_label: dict[str, list[str]] = field(default_factory=lambda: {})
    label_ids_by_paragraph: dict[str, list[str]] = field(default_factory=lambda: {})
    concept_ids_by_paragraph: dict[str, list[str]] = field(default_factory=lambda: {})
    paragraph_ids_by_concept: dict[str, list[str]] = field(default_factory=lambda: {})
    label_display_names_by_id: dict[str, str] = field(default_factory=lambda: {})
    label_concept_ids_by_id: dict[str, list[str]] = field(default_factory=lambda: {})
    concept_texts_by_id: dict[str, str] = field(default_factory=lambda: {})


def build_corpus_index(result: LabelGenerationResult) -> CorpusIndex:
    """Build retrieval-ready paragraph-side artifacts from a fitted result."""

    concept_text_by_id = {concept.id: concept.normalized for concept in result.concepts}
    label_display_names_by_id = {
        community.id: community.display_name for community in result.communities
    }
    label_concept_ids_by_id = {
        community.id: sorted(community.concept_ids) for community in result.communities
    }

    concept_ids_by_paragraph_sets: dict[str, set[str]] = defaultdict(set)
    for mention in result.mentions:
        concept_ids_by_paragraph_sets[mention.paragraph_id].add(mention.concept_id)

    label_ids_by_paragraph: dict[str, list[str]] = {
        paragraph_labels.paragraph_id: list(paragraph_labels.label_ids)
        for paragraph_labels in result.paragraph_labels
    }

    paragraph_ids_by_label_sets: dict[str, set[str]] = defaultdict(set)
    for paragraph_id, label_ids in label_ids_by_paragraph.items():
        for label_id in label_ids:
            paragraph_ids_by_label_sets[label_id].add(paragraph_id)

    paragraph_ids_by_concept_sets: dict[str, set[str]] = defaultdict(set)
    for paragraph_id, concept_ids in concept_ids_by_paragraph_sets.items():
        for concept_id in concept_ids:
            paragraph_ids_by_concept_sets[concept_id].add(paragraph_id)

    paragraphs_by_id: dict[str, IndexedParagraph] = {}
    concept_ids_by_paragraph: dict[str, list[str]] = {}
    for paragraph in result.paragraphs:
        concept_ids = sorted(concept_ids_by_paragraph_sets.get(paragraph.id, set()))
        label_ids = sorted(label_ids_by_paragraph.get(paragraph.id, []))
        concept_ids_by_paragraph[paragraph.id] = concept_ids
        paragraphs_by_id[paragraph.id] = IndexedParagraph(
            paragraph_id=paragraph.id,
            text=paragraph.text,
            metadata=paragraph.metadata,
            concept_ids=concept_ids,
            concept_texts=[
                concept_text_by_id[concept_id]
                for concept_id in concept_ids
                if concept_id in concept_text_by_id
            ],
            label_ids=label_ids,
            label_display_names=[
                label_display_names_by_id[label_id]
                for label_id in label_ids
                if label_id in label_display_names_by_id
            ],
        )

    return CorpusIndex(
        paragraphs_by_id=paragraphs_by_id,
        paragraph_ids_by_label={
            label_id: sorted(paragraph_ids)
            for label_id, paragraph_ids in paragraph_ids_by_label_sets.items()
        },
        label_ids_by_paragraph=label_ids_by_paragraph,
        concept_ids_by_paragraph=concept_ids_by_paragraph,
        paragraph_ids_by_concept={
            concept_id: sorted(paragraph_ids)
            for concept_id, paragraph_ids in paragraph_ids_by_concept_sets.items()
        },
        label_display_names_by_id=label_display_names_by_id,
        label_concept_ids_by_id=label_concept_ids_by_id,
        concept_texts_by_id=concept_text_by_id,
    )
