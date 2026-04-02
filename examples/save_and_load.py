"""Pipeline persistence example for labelrag."""

import shutil
from pathlib import Path

from labelrag import RAGPipeline, RAGPipelineConfig


def main() -> None:
    """Fit, save, load, and reuse a pipeline with compressed persistence."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False

    pipeline = RAGPipeline(config)
    pipeline.fit(
        [
            "OpenAI builds language models.",
            "OpenAI deploys language models in production.",
        ]
    )

    output_path = Path("rag-pipeline-example")
    pipeline.save(output_path, format="json.gz")

    loaded = RAGPipeline.load(output_path)
    retrieval = loaded.build_context("How does OpenAI use language models?")

    print(f"Saved pipeline to: {output_path}")
    print(f"Reloaded prompt context:\n{retrieval.prompt_context}")
    print(f"Reloaded metadata: {retrieval.metadata}")

    shutil.rmtree(output_path, ignore_errors=True)


if __name__ == "__main__":
    main()
