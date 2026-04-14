"""Compare label-free fallback policies in labelrag."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import RAGPipeline, RAGPipelineConfig

PARAGRAPHS = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]

QUESTION = "monitoring and starship reactors"


def build_pipeline(strategy: str) -> RAGPipeline:
    """Construct one pipeline configured for a specific fallback strategy."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.retrieval.label_free_fallback_strategy = strategy

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(PARAGRAPHS)
    return pipeline


def main() -> None:
    """Run the same no-label query against all fallback strategies."""

    for strategy in (
        "concept_overlap_only",
        "concept_overlap_semantic_rerank",
        "semantic_only",
    ):
        pipeline = build_pipeline(strategy)
        result = pipeline.build_context(QUESTION)

        print(f"\n=== {strategy} ===")
        print(f"retrieval_strategy: {result.metadata['retrieval_strategy']}")
        print(f"semantic_reranking_enabled: {result.metadata['semantic_reranking_enabled']}")
        print(f"retrieved_paragraph_count: {len(result.retrieved_paragraphs)}")
        for paragraph in result.retrieved_paragraphs:
            print(
                f"- {paragraph.paragraph_id}: "
                f"concept_overlap={paragraph.concept_overlap_count} "
                f"semantic_similarity={paragraph.semantic_similarity}"
            )


if __name__ == "__main__":
    main()
