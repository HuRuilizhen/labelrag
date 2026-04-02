# labelrag

`labelrag` is a Python library for label-driven retrieval-augmented generation
pipelines built on top of `paralabelgen`.

- PyPI distribution: `labelrag`
- Python import package: `labelrag`
- Core dependency target: `paralabelgen==0.2.0`
- Default extraction path: spaCy via `paralabelgen`

## Install

```bash
pip install labelrag
```

If you want to use the default spaCy-backed labeling path, install a compatible
English pipeline such as:

```bash
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is the recommended default model, but you can point the
underlying `LabelGeneratorConfig` at another installed compatible spaCy
pipeline.

## Quick Start

### Retrieval-only workflow

```python
from labelrag import RAGPipeline, RAGPipelineConfig

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]

pipeline = RAGPipeline(RAGPipelineConfig())
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

pipeline = RAGPipeline(RAGPipelineConfig())
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

The current retrieval layer is deterministic and label-driven.

- `fit(...)` delegates paragraph analysis to `labelgen.LabelGenerator`
- `build_context(...)` maps the question into the fitted label space
- retrieval uses greedy coverage over query label IDs
- label-free queries can fall back to deterministic concept overlap
- `require_full_label_coverage=True` suppresses partial retrieval results while
  preserving attempted coverage trace in metadata

Tie-break order for greedy retrieval is:

1. larger overlap with remaining query labels
2. larger overlap on query concept IDs
3. larger total paragraph label count
4. lexicographically smaller `paragraph_id`

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
- `IndexedParagraph`, `QueryAnalysis`, `RetrievedParagraph`
- `RetrievalResult`, `RAGAnswerResult`
- `GeneratedAnswer`, `AnswerGenerator`
- `OpenAICompatibleAnswerGenerator`, `OpenAICompatibleConfig`
- convenience re-export: `Paragraph`

Detailed API notes are available in [`docs/public_api.md`](docs/public_api.md).

## Examples

Runnable examples are available in [`examples/`](examples/):

- [`examples/basic_usage.py`](examples/basic_usage.py)
- [`examples/custom_config.py`](examples/custom_config.py)
- [`examples/save_and_load.py`](examples/save_and_load.py)
- [`examples/provider_answer.py`](examples/provider_answer.py)

## Persistence Notes

`save(path)` produces a human-inspectable directory containing:

- `config.json`
- `label_generator.json`
- `corpus_index.json`
- `fit_result.json`

The persistence layer now supports:

- `json`
- `json.gz`

Compression is applied to the full saved snapshot rather than mixing compressed
and uncompressed artifacts in one directory.

Public guarantee:

- a saved and reloaded pipeline should preserve retrieval behavior for the same
  fitted state, question, and config

Current update boundary:

- `fit(...)` is batch-only
- adding new paragraphs currently requires a full refit
- save/load restores a static fitted state rather than an incrementally
  updateable corpus state

## Configuration Notes

- `RetrievalConfig.max_paragraphs` sets the hard retrieval limit
- `RetrievalConfig.allow_label_free_fallback` enables deterministic concept
  overlap fallback for label-free queries
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
