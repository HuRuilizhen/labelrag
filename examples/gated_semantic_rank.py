"""Compare semantic-first gated retrieval strategies in labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import RAGPipeline, RAGPipelineConfig

PARAGRAPHS = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]


def build_pipeline() -> RAGPipeline:
    """Construct one pipeline for the gated semantic-first examples."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(PARAGRAPHS)
    return pipeline


def run_main_path_example() -> None:
    """Show the semantic-first gated strategy on the main retrieval path."""

    pipeline = build_pipeline()
    pipeline.config.retrieval.retrieval_strategy = "label_gate_semantic_rank"
    result = pipeline.build_context("How do developers use language models?")

    print("\n=== main path: label_gate_semantic_rank ===")
    print(f"retrieval_strategy: {result.metadata['retrieval_strategy']}")
    print(f"semantic_reranking_enabled: {result.metadata['semantic_reranking_enabled']}")
    for paragraph in result.retrieved_paragraphs:
        print(
            f"- {paragraph.paragraph_id}: "
            f"semantic_similarity={paragraph.semantic_similarity} "
            f"marginal_gain={paragraph.marginal_gain}"
        )


def run_fallback_example() -> None:
    """Show the semantic-first gated strategy on the fallback path."""

    pipeline = build_pipeline()
    pipeline.config.retrieval.label_free_fallback_strategy = "concept_gate_semantic_rank"
    result = pipeline.build_context("monitoring and starship reactors")

    print("\n=== fallback: concept_gate_semantic_rank ===")
    print(f"retrieval_strategy: {result.metadata['retrieval_strategy']}")
    print(
        "label_free_fallback_strategy: "
        f"{result.metadata['label_free_fallback_strategy']}"
    )
    print(f"semantic_reranking_enabled: {result.metadata['semantic_reranking_enabled']}")
    for paragraph in result.retrieved_paragraphs:
        print(
            f"- {paragraph.paragraph_id}: "
            f"semantic_similarity={paragraph.semantic_similarity} "
            f"concept_overlap={paragraph.concept_overlap_count}"
        )


def main() -> None:
    """Run both gated semantic-first examples."""

    run_main_path_example()
    run_fallback_example()


if __name__ == "__main__":
    main()
