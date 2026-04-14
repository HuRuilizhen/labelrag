"""Public package exports for the `labelrag` package."""

from labelgen import Paragraph

from labelrag.config import EmbeddingConfig, PromptConfig, RAGPipelineConfig, RetrievalConfig
from labelrag.embedding.provider import EmbeddingProvider
from labelrag.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from labelrag.generation.generator import AnswerGenerator, GeneratedAnswer
from labelrag.generation.openai_compatible import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
)
from labelrag.pipeline.rag_pipeline import RAGPipeline
from labelrag.types import (
    ConceptRecord,
    IndexedParagraph,
    LabelRecord,
    QueryAnalysis,
    RAGAnswerResult,
    RetrievalResult,
    RetrievedParagraph,
)

__all__ = [
    "AnswerGenerator",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "ConceptRecord",
    "GeneratedAnswer",
    "IndexedParagraph",
    "LabelRecord",
    "OpenAICompatibleAnswerGenerator",
    "OpenAICompatibleConfig",
    "Paragraph",
    "PromptConfig",
    "QueryAnalysis",
    "RAGAnswerResult",
    "RAGPipeline",
    "RAGPipelineConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievedParagraph",
    "SentenceTransformerEmbeddingProvider",
]
