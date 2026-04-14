# labelrag

`labelrag` is a Python library for label-driven retrieval-augmented generation
pipelines built on top of `paralabelgen`.

- PyPI distribution: `labelrag`
- Python import package: `labelrag`
- Core dependency target: `paralabelgen==0.2.2`
- Primary supported extraction path: `paralabelgen` LLM concept extraction
- First semantic-reranking embedding provider: `sentence-transformers`

## Install

```bash
pip install labelrag
```

If you want to use the spaCy-backed extraction path, install a compatible
English pipeline such as:

```bash
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is a convenient local option, but `0.1.0` release validation
targets the `paralabelgen==0.2.2` LLM concept-extraction pipeline as the
primary supported extraction path.

The first shipped semantic-reranking provider uses `sentence-transformers`.
Its model weights may be downloaded on first use if they are not already cached
locally.

## Quick Start

### Retrieval-only workflow

```python
from labelrag import (
    RAGPipeline,
    RAGPipelineConfig,
)

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]

config = RAGPipelineConfig()
config.labelgen.extractor_mode = "heuristic"
config.labelgen.use_graph_community_detection = False

pipeline = RAGPipeline(config)
pipeline.fit(paragraphs)

retrieval = pipeline.build_context("How do developers use language models?")
print(retrieval.prompt_context)
print(retrieval.metadata)
```

### Retrieval plus provider-backed answer generation

```python
from labelrag import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
    RAGPipeline,
    RAGPipelineConfig,
)

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]

config = RAGPipelineConfig()
config.labelgen.extractor_mode = "heuristic"
config.labelgen.use_graph_community_detection = False

pipeline = RAGPipeline(config)
pipeline.fit(paragraphs)

generator = OpenAICompatibleAnswerGenerator(
    OpenAICompatibleConfig(
        model="mistral-small-latest",
        api_key_env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
    )
)

answer = pipeline.answer_with_generator(
    "How do developers use language models?",
    generator,
)
print(answer.answer_text)
print(answer.metadata)
```

## Retrieval Model

The current retrieval layer is deterministic and still label-driven at the
candidate-generation stage.

- `fit(...)` delegates paragraph analysis to `labelgen.LabelGenerator`
- `fit(...)` also builds paragraph embeddings
- `build_context(...)` maps the question into the fitted label space
- retrieval uses greedy coverage over query label IDs
- semantic similarity is used as a secondary ranking signal inside greedy
  selection
- label-free queries can use configurable fallback strategies
- `require_full_label_coverage=True` suppresses partial retrieval results while
  preserving attempted coverage trace in metadata

Greedy selection order is:

1. larger overlap with remaining query labels
2. larger semantic similarity
3. larger overlap on query concept IDs
4. larger total paragraph label count
5. lexicographically smaller `paragraph_id`

`0.1.1` supports three label-free fallback strategies:

- `concept_overlap_only`
- `concept_overlap_semantic_rerank`
- `semantic_only`

The default is `concept_overlap_semantic_rerank`.

## OpenAI-Compatible Provider Notes

The built-in answer-generation adapter targets a minimal OpenAI-compatible
chat-completions API surface.

It supports:

- standard base URLs such as `https://api.openai.com/v1`
- full endpoint URLs such as `https://api.mistral.ai/v1/chat/completions`
- API key injection through explicit config or optional environment-variable
  lookup
- non-streaming text generation for `answer_with_generator(...)`

This adapter is intended to cover providers such as OpenAI, Mistral, and Qwen
when they expose an OpenAI-compatible endpoint shape.

## Public API

The main public entrypoints are:

- `RAGPipeline`
- `RAGPipelineConfig`, `RetrievalConfig`, `PromptConfig`
- `IndexedParagraph`, `LabelRecord`, `ConceptRecord`
- `QueryAnalysis`, `RetrievedParagraph`
- `RetrievalResult`, `RAGAnswerResult`
- `GeneratedAnswer`, `AnswerGenerator`
- `OpenAICompatibleAnswerGenerator`, `OpenAICompatibleConfig`
- convenience re-export: `Paragraph`

`RAGPipeline` also exposes record-oriented inspection helpers for
paragraph/label/concept lookup workflows:

- `get_paragraph(...)`
- `get_label(...)`
- `get_paragraph_labels(...)`
- `get_paragraph_concepts(...)`
- `get_label_paragraphs(...)`
- `get_concept_paragraphs(...)`

Lower-level ID-oriented helpers remain available when you only need stable IDs:

- `get_label_paragraph_ids(...)`
- `get_paragraph_label_ids(...)`
- `get_paragraph_concept_ids(...)`
- `get_concept_paragraph_ids(...)`

Detailed API notes are available in [`docs/public_api.md`](docs/public_api.md).

## Embedding Notes

- `RAGPipeline.fit(...)` now requires an embedding provider
- the default runtime path is `RAGPipeline(config)` and resolves the embedding
  provider from `config.embedding`
- explicit `embedding_provider=` is still available as an advanced override
- the first shipped provider is `SentenceTransformerEmbeddingProvider`
- the default model is `sentence-transformers/all-MiniLM-L6-v2`
- the model may be downloaded on first use
- offline environments should pre-cache the embedding model before running
  `fit(...)`

Common runtime failures:

- missing `sentence-transformers` package:
  - reinstall project dependencies, for example `pip install -e .`
- model load/download failure:
  - verify the configured model name
  - ensure the model is already cached locally or that the environment can
    reach Hugging Face

## Examples

Runnable examples are available in [`examples/`](examples/):

- [`examples/basic_usage.py`](examples/basic_usage.py)
- [`examples/custom_config.py`](examples/custom_config.py)
- [`examples/inspection_api.py`](examples/inspection_api.py)
- [`examples/fallback_policies.py`](examples/fallback_policies.py)
- [`examples/semantic_rerank.py`](examples/semantic_rerank.py)
- [`examples/save_and_load.py`](examples/save_and_load.py)
- [`examples/provider_answer.py`](examples/provider_answer.py)

Example note:

- the runnable example scripts use a tiny local demo embedding provider so they
  stay runnable offline
- production usage should prefer `SentenceTransformerEmbeddingProvider`

## Persistence Notes

`save(path)` produces a human-inspectable directory containing:

- `manifest.json`
- `config.json`
- `label_generator.json`
- `corpus_index.json`
- `fit_result.json`
- `paragraph_embeddings.npz`

The persistence layer now supports:

- `json`
- `json.gz`

Compression is applied to the full saved snapshot rather than mixing compressed
and uncompressed artifacts in one directory.

Snapshots written by the current release include a lightweight manifest
describing the saved version, persistence format, and expected artifacts.

Public guarantee:

- a saved and reloaded pipeline should preserve retrieval behavior for the same
  fitted state, question, and config

Current update boundary:

- `fit(...)` is batch-only
- adding new paragraphs currently requires a full refit
- save/load restores a static fitted state rather than an incrementally
  updateable corpus state

Legacy snapshot note:

- loading pre-embedding snapshots remains a best-effort compatibility path
- when older snapshots are missing derived concept inspection tables, `load()`
  may rebuild them from paragraph-side concept data that is still present
- when older snapshots predate `paragraph_embeddings.npz`, `load()` may rebuild
  paragraph embeddings from persisted paragraph texts if an embedding provider
  is available
- persisted manifests include a non-empty `labelrag_version`
- `save()` fails explicitly if the current package version cannot be determined
  for manifest writing

## Configuration Notes

- `RetrievalConfig.max_paragraphs` sets the hard retrieval limit
- `RetrievalConfig.allow_label_free_fallback` enables deterministic concept
  fallback behavior for label-free queries
- `RetrievalConfig.label_free_fallback_strategy` selects one of:
  - `concept_overlap_only`
  - `concept_overlap_semantic_rerank`
  - `semantic_only`
- `RetrievalConfig.require_full_label_coverage` suppresses partial retrieval
  output when not all query labels can be covered
- `PromptConfig.include_paragraph_ids` includes stable paragraph IDs in the
  rendered prompt context
- `PromptConfig.include_label_annotations` includes paragraph label annotations
  in rendered prompt context
- `PromptConfig.max_context_characters` applies a hard cap to rendered context
  length

## Development Checks

```bash
.venv/bin/ruff check . --fix
.venv/bin/pyright
.venv/bin/pytest
```

## Release Checks

```bash
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```
