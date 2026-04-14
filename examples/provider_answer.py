"""Provider-backed answer generation example for labelrag.

Configure the provider through environment variables before running:

    export LABELRAG_LLM_MODEL=mistral-small-latest
    export LABELRAG_LLM_BASE_URL=https://api.mistral.ai/v1
    export LABELRAG_LLM_API_KEY_ENV_VAR=MISTRAL_API_KEY
    export MISTRAL_API_KEY=...
    python examples/provider_answer.py
"""

from __future__ import annotations

import os

from _demo_embedding import DemoEmbeddingProvider

from labelrag import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
    RAGPipeline,
    RAGPipelineConfig,
)


def main() -> None:
    """Fit a pipeline and request a real provider-backed answer."""

    model = os.environ.get("LABELRAG_LLM_MODEL", "")
    if not model:
        print("Skipping provider example: set LABELRAG_LLM_MODEL to run a real request.")
        return

    base_url = os.environ.get("LABELRAG_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key_env_var = os.environ.get("LABELRAG_LLM_API_KEY_ENV_VAR", "OPENAI_API_KEY")
    if not os.environ.get(api_key_env_var):
        print(
            "Skipping provider example: "
            f"set `{api_key_env_var}` before running a real provider request."
        )
        return

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

    generator = OpenAICompatibleAnswerGenerator(
        OpenAICompatibleConfig(
            model=model,
            base_url=base_url,
            api_key_env_var=api_key_env_var,
        )
    )
    answer = pipeline.answer_with_generator(
        "How do developers use language models?",
        generator,
    )

    print(f"Model: {model}")
    print(f"Answer: {answer.answer_text}")
    print(f"Metadata: {answer.metadata}")


if __name__ == "__main__":
    main()
