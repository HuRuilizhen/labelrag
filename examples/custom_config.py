"""Custom retrieval configuration example for labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import (
    RAGPipeline,
    RAGPipelineConfig,
)


def main() -> None:
    """Run retrieval with explicit deterministic configuration changes."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.retrieval.max_paragraphs = 1
    config.retrieval.allow_label_free_fallback = False
    config.prompt.include_label_annotations = True

    paragraphs = [
        "OpenAI builds language models for developers.",
        "Developers use language models in production systems.",
        "Production systems rely on monitoring and deployment tooling.",
    ]

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(paragraphs)

    labeled = pipeline.build_context("Developers use language models in production systems.")
    label_free = pipeline.build_context("Quantum batteries improve starship reactors.")

    print("Labeled query with prompt label annotations:")
    print(labeled.prompt_context)
    print(f"- metadata={labeled.metadata}")

    print("\nLabel-free query with fallback disabled:")
    print(f"- prompt_context_present={bool(label_free.prompt_context)}")
    print(f"- metadata={label_free.metadata}")


if __name__ == "__main__":
    main()
