"""Basic end-to-end retrieval example for labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import (
    RAGPipeline,
    RAGPipelineConfig,
)


def main() -> None:
    """Fit a small corpus and inspect retrieval outputs."""

    paragraphs = [
        "OpenAI builds language models for developers.",
        "Developers use language models in production systems.",
        "Production systems need monitoring and evaluation tooling.",
    ]

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(paragraphs)

    result = pipeline.build_context("How do developers use language models?")

    print("Query Analysis:")
    print(f"- concepts={result.query_analysis.concepts}")
    print(f"- label_ids={result.query_analysis.label_ids}")
    print("\nRetrieved Paragraphs:")
    for paragraph in result.retrieved_paragraphs:
        print(
            f"- {paragraph.paragraph_id}: "
            f"matched_labels={paragraph.matched_label_ids} "
            f"new_labels={paragraph.newly_covered_label_ids}"
        )

    print("\nPrompt Context:")
    print(result.prompt_context)
    print("\nMetadata:")
    print(result.metadata)


if __name__ == "__main__":
    main()
