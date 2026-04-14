"""Public configuration models for `labelrag`."""

from dataclasses import dataclass, field

from labelgen import LabelGeneratorConfig


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for paragraph/query embedding behavior."""

    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int | None = None
    normalize: bool = True


@dataclass(slots=True)
class RetrievalConfig:
    """Configuration for paragraph retrieval behavior."""

    max_paragraphs: int = 8
    require_full_label_coverage: bool = False
    allow_label_free_fallback: bool = True
    label_free_fallback_strategy: str = "concept_overlap_semantic_rerank"


@dataclass(slots=True)
class PromptConfig:
    """Configuration for prompt context rendering."""

    include_paragraph_ids: bool = True
    include_label_annotations: bool = False
    max_context_characters: int | None = None


@dataclass(slots=True)
class RAGPipelineConfig:
    """Top-level public configuration for `RAGPipeline`."""

    labelgen: LabelGeneratorConfig = field(default_factory=LabelGeneratorConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
