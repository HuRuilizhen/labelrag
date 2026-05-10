"""Demonstrate greedy label coverage plus semantic backfill."""

from _demo_embedding import DemoEmbeddingProvider

from labelrag import RAGPipeline, RAGPipelineConfig

PARAGRAPHS = [
    "Developers use language models in production systems.",
    "Developers use language models in real products.",
    "Developers use language models across product teams.",
]


def main() -> None:
    """Run one retrieval that triggers semantic backfill."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.retrieval.max_paragraphs = 3

    pipeline = RAGPipeline(config, embedding_provider=DemoEmbeddingProvider())
    pipeline.fit(PARAGRAPHS)

    result = pipeline.build_context("How do developers use language models?")

    print("=== greedy_label_coverage_semantic_rerank with semantic backfill ===")
    print(f"retrieval_strategy: {result.metadata['retrieval_strategy']}")
    print(f"semantic_backfill_used: {result.metadata['semantic_backfill_used']}")
    for paragraph in result.retrieved_paragraphs:
        print(
            f"- {paragraph.paragraph_id}: "
            f"retrieval_score={paragraph.retrieval_score} "
            f"retrieval_score_kind={paragraph.retrieval_score_kind} "
            f"semantic_similarity={paragraph.semantic_similarity}"
        )


if __name__ == "__main__":
    main()
