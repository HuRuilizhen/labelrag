"""Record-oriented inspection API example for labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import (
    RAGPipeline,
    RAGPipelineConfig,
)


def main() -> None:
    """Fit a small corpus and inspect paragraph, label, and concept records."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    assert pipeline.corpus_index is not None
    paragraph_id = sorted(pipeline.corpus_index.paragraphs_by_id)[0]
    paragraph = pipeline.get_paragraph(paragraph_id)
    assert paragraph is not None

    print("Paragraph:")
    print(f"- id={paragraph.paragraph_id}")
    print(f"- labels={paragraph.label_ids}")
    print(f"- concepts={paragraph.concept_ids}")

    if paragraph.label_ids:
        label_id = paragraph.label_ids[0]
        label = pipeline.get_label(label_id)
        assert label is not None
        print("\nLabel record:")
        print(f"- label_id={label.label_id}")
        print(f"- display_name={label.display_name}")
        print(f"- concept_ids={label.concept_ids}")
        print(f"- paragraph_ids={label.paragraph_ids}")
        print("\nParagraph label records:")
        for paragraph_label in pipeline.get_paragraph_labels(paragraph_id):
            print(
                f"- {paragraph_label.label_id}: "
                f"display_name={paragraph_label.display_name} "
                f"paragraph_ids={paragraph_label.paragraph_ids}"
            )

    if paragraph.concept_ids:
        concept_id = paragraph.concept_ids[0]
        print("\nParagraph concept records:")
        for concept in pipeline.get_paragraph_concepts(paragraph_id):
            print(
                f"- {concept.concept_id}: "
                f"text={concept.text} "
                f"paragraph_ids={concept.paragraph_ids}"
            )
        print("\nConcept paragraph records:")
        for concept_paragraph in pipeline.get_concept_paragraphs(concept_id):
            print(f"- {concept_paragraph.paragraph_id}: {concept_paragraph.text}")


if __name__ == "__main__":
    main()
