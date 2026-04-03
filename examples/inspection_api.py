"""Inspection API example for labelrag."""

from labelrag import RAGPipeline, RAGPipelineConfig


def main() -> None:
    """Fit a small corpus and inspect paragraph/label/concept lookups."""

    pipeline = RAGPipeline(RAGPipelineConfig())
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
        print("\nLabel lookup:")
        print(f"- label_id={label_id}")
        print(f"- paragraph_ids={pipeline.get_label_paragraph_ids(label_id)}")

    if paragraph.concept_ids:
        concept_id = paragraph.concept_ids[0]
        print("\nConcept lookup:")
        print(f"- concept_id={concept_id}")
        print(f"- paragraph_ids={pipeline.get_concept_paragraph_ids(concept_id)}")


if __name__ == "__main__":
    main()
