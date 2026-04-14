"""Semantic reranking example for labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import (
    RAGPipeline,
    RAGPipelineConfig,
)


def main() -> None:
    """Show semantic similarity in the main greedy retrieval path."""

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

    result = pipeline.build_context("How do developers use language models?")

    print("Retrieval strategy:")
    print(f"- {result.metadata['retrieval_strategy']}")
    print("\nRetrieved paragraphs:")
    for paragraph in result.retrieved_paragraphs:
        print(
            f"- {paragraph.paragraph_id}: "
            f"semantic_similarity={paragraph.semantic_similarity} "
            f"marginal_gain={paragraph.marginal_gain}"
        )


if __name__ == "__main__":
    main()
