# Public API

This document defines the intended public API for `labelrag`.

The goal is to expose a small, stable surface for label-driven RAG evaluation
workflows while keeping `labelgen` as the only dependency responsible for
concept extraction, community detection, and label assignment.

Current dependency target:

- `paralabelgen==0.2.0`

## Package Entry Point

Recommended import style:

```python
from labelrag import (
    ConceptRecord,
    GeneratedAnswer,
    IndexedParagraph,
    LabelRecord,
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
    PromptConfig,
    QueryAnalysis,
    RAGAnswerResult,
    RAGPipeline,
    RAGPipelineConfig,
    RetrievalConfig,
    RetrievalResult,
    RetrievedParagraph,
)
```

`labelrag` should also re-export `Paragraph` from `labelgen` for convenience:

```python
from labelrag import Paragraph
```

## Main Pipeline

### `RAGPipeline`

Primary public entrypoint for corpus fitting, query analysis, retrieval, and
optional answer generation.

Constructor:

```python
RAGPipeline(
    config: RAGPipelineConfig | None = None,
    generator: AnswerGenerator | None = None,
)
```

Behavior:

- `config` defaults to `RAGPipelineConfig()`
- `generator` is optional
- when no generator is configured, `answer(...)` may return an empty
  `answer_text`

### Methods

#### `fit`

```python
fit(paragraphs: list[str] | list[Paragraph]) -> RAGPipeline
```

Fits the underlying `labelgen.LabelGenerator`, builds paragraph-side retrieval
artifacts, and stores enough state for later `build_context(...)`, `answer(...)`,
and `save(...)`.

Accepted inputs:

- `list[str]`
- `list[labelgen.Paragraph]`

Current update boundary:

- `fit(...)` is batch-only
- the current package does not support incremental paragraph ingestion
- adding new paragraphs currently requires a full refit
- `save(...)` and `load(...)` restore a static fitted state rather than an
  incrementally updateable corpus state

#### `build_context`

```python
build_context(question: str) -> RetrievalResult
```

Runs query-side analysis and retrieval only. Does not call the answer
generator.

#### Inspection lookup methods

```python
get_paragraph(paragraph_id: str) -> IndexedParagraph | None
get_label(label_id: str) -> LabelRecord | None
get_paragraph_labels(paragraph_id: str) -> list[LabelRecord]
get_paragraph_concepts(paragraph_id: str) -> list[ConceptRecord]
get_concept_paragraphs(concept_id: str) -> list[IndexedParagraph]
get_label_paragraph_ids(label_id: str) -> list[str]
get_label_paragraphs(label_id: str) -> list[IndexedParagraph]
get_paragraph_label_ids(paragraph_id: str) -> list[str]
get_paragraph_concept_ids(paragraph_id: str) -> list[str]
get_concept_paragraph_ids(concept_id: str) -> list[str]
```

Behavior:

- all inspection methods require a fitted pipeline
- `get_paragraph(...)` returns `None` for an unknown paragraph ID
- `get_label(...)` returns `None` for an unknown label ID
- list-returning inspection methods return `[]` for unknown IDs
- returned list order is deterministic

Inspection API layering:

- record-oriented inspection APIs are the primary inspection surface:
  - `get_paragraph(...)`
  - `get_label(...)`
  - `get_paragraph_labels(...)`
  - `get_paragraph_concepts(...)`
  - `get_label_paragraphs(...)`
  - `get_concept_paragraphs(...)`
- ID-oriented helper APIs remain available as lower-level helpers:
  - `get_label_paragraph_ids(...)`
  - `get_paragraph_label_ids(...)`
  - `get_paragraph_concept_ids(...)`
  - `get_concept_paragraph_ids(...)`

#### `answer`

```python
answer(question: str) -> RAGAnswerResult
```

Runs query analysis, retrieval, prompt context construction, and optional answer
generation using the pipeline-level default generator configured at
construction time.

Behavior:

- if a pipeline-level generator is configured, `answer(...)` uses it
- if no generator is configured, `answer(...)` still returns retrieval outputs
  and prompt context, but `answer_text` may be empty

#### `answer_with_generator`

```python
answer_with_generator(
    question: str,
    generator: AnswerGenerator,
) -> RAGAnswerResult
```

Runs the full answer flow with a per-call generator override.

Behavior:

- the passed generator overrides the pipeline-level default generator for this
  call only
- this is the explicit way to switch provider, model, or credentials without
  rebuilding the pipeline

#### `save`

```python
save(
    path: str | Path,
    format: Literal["json", "json.gz"] | None = None,
) -> None
```

Persists pipeline configuration, fitted `LabelGenerator` state, and retrieval
artifacts.

Behavior:

- when `format` is omitted, the implementation auto-detects an existing
  persistence layout in the target directory
- when `format` is explicit, it overrides auto-detection
- the current release supports whole-snapshot persistence in either:
  - `json`
  - `json.gz`
- mixed compressed and uncompressed artifact layouts are out of scope
- snapshots written by the current release include a lightweight manifest
  artifact:
  - `manifest.json`
  - `manifest.json.gz`

#### `load`

```python
RAGPipeline.load(
    path: str | Path,
    format: Literal["json", "json.gz"] | None = None,
) -> RAGPipeline
```

Loads a previously saved pipeline.

Behavior:

- when `format` is omitted, the loader auto-detects `json` versus `json.gz`
- when `format` is explicit, the loader does not guess and requires the chosen
  format to exist completely
- snapshots written by the current release are expected to include a manifest
- loading pre-`0.0.2` snapshots without a manifest remains supported on a
  best-effort basis
- when legacy snapshots are missing derived concept inspection tables, the
  loader may rebuild them from stored paragraph-side concept assignments

### Error Conditions

The following conditions should raise explicit runtime errors:

- `build_context(...)` before `fit(...)`
- `answer(...)` before `fit(...)`
- `save(...)` before the pipeline has enough state to serialize

## Configuration Models

All public config models should use `@dataclass(slots=True)`.

### `RetrievalConfig`

```python
@dataclass(slots=True)
class RetrievalConfig:
    max_paragraphs: int = 8
    require_full_label_coverage: bool = False
    allow_label_free_fallback: bool = True
```

Field meaning:

- `max_paragraphs`: maximum retrieved paragraphs returned in one result
- `require_full_label_coverage`: when `True`, the pipeline may signal incomplete
  retrieval more strictly if not all query labels are covered
- `allow_label_free_fallback`: whether a query with no assigned labels may still
  return a fallback retrieval result

### `PromptConfig`

```python
@dataclass(slots=True)
class PromptConfig:
    include_paragraph_ids: bool = True
    include_label_annotations: bool = False
    max_context_characters: int | None = None
```

Field meaning:

- `include_paragraph_ids`: include stable paragraph IDs in rendered context
- `include_label_annotations`: optionally include label metadata in rendered
  prompt context
- `max_context_characters`: optional hard limit for prompt context length

### `RAGPipelineConfig`

```python
@dataclass(slots=True)
class RAGPipelineConfig:
    labelgen: LabelGeneratorConfig = field(default_factory=LabelGeneratorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
```

Boundary rule:

- `labelrag` may configure `LabelGeneratorConfig`
- `labelrag` should not patch `labelgen` internals directly

## Data Models

All public result models should use `@dataclass(slots=True)`.

### `IndexedParagraph`

```python
@dataclass(slots=True)
class IndexedParagraph:
    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    concept_ids: list[str]
    concept_texts: list[str]
    label_ids: list[str]
    label_display_names: list[str]
```

Meaning:

- normalized paragraph plus the label-side metadata needed for retrieval and
  trace inspection

### `LabelRecord`

```python
@dataclass(slots=True)
class LabelRecord:
    label_id: str
    display_name: str
    concept_ids: list[str]
    paragraph_ids: list[str]
```

Meaning:

- lightweight label-side inspection record built from fitted retrieval state
- intended for direct inspection workflows rather than retrieval scoring

### `ConceptRecord`

```python
@dataclass(slots=True)
class ConceptRecord:
    concept_id: str
    text: str
    paragraph_ids: list[str]
```

Meaning:

- lightweight concept-side inspection record built from fitted retrieval state
- intended for direct inspection workflows rather than query analysis output

### `QueryAnalysis`

```python
@dataclass(slots=True)
class QueryAnalysis:
    query_text: str
    concepts: list[str]
    concept_ids: list[str]
    label_ids: list[str]
    label_display_names: list[str]
```

Meaning:

- structured query-side output derived from
  `labelgen.LabelGenerator.transform([question])`

### `RetrievedParagraph`

```python
@dataclass(slots=True)
class RetrievedParagraph:
    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    newly_covered_label_ids: list[str]
    already_covered_label_ids: list[str]
    matched_label_ids: list[str]
    matched_concept_ids: list[str]
    paragraph_label_ids: list[str]
    paragraph_concept_ids: list[str]
    concept_overlap_count: int
    marginal_gain: int
    retrieval_score: float
```

Meaning:

- one retrieval step record with enough data to audit why the paragraph was
  selected
- `newly_covered_label_ids` records the labels this retrieval step contributes
  toward final coverage
- `already_covered_label_ids` records labels that matched but were already
  covered by earlier retrieved paragraphs
- `concept_overlap_count` records the number of overlapping query concept IDs
  used in tie-break decisions

### `RetrievalResult`

```python
@dataclass(slots=True)
class RetrievalResult:
    question: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any]
```

Minimum required metadata keys:

- `covered_label_ids`
- `uncovered_label_ids`
- `attempted_covered_label_ids`
- `attempted_uncovered_label_ids`
- `retrieval_strategy`
- `used_label_free_fallback`
- `require_full_label_coverage`
- `full_label_coverage_met`

Recommended additional metadata keys:

- `query_label_ids`
- `retrieval_limit`

### `GeneratedAnswer`

```python
@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    metadata: dict[str, Any]
```

Meaning:

- generator output normalized into a simple, model-agnostic structure

### `RAGAnswerResult`

```python
@dataclass(slots=True)
class RAGAnswerResult:
    question: str
    answer_text: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any]
```

Minimum required metadata keys:

- `covered_label_ids`
- `uncovered_label_ids`
- `attempted_covered_label_ids`
- `attempted_uncovered_label_ids`
- `retrieval_strategy`
- `used_label_free_fallback`
- `require_full_label_coverage`
- `full_label_coverage_met`
- `generator_name`
- `generation_model`

Recommended additional metadata keys:

- `query_label_ids`
- `retrieval_limit`
- `generation_metadata`

## Built-In Provider Adapter

The package also exports a minimal built-in OpenAI-compatible answer generator.

### `OpenAICompatibleConfig`

```python
@dataclass(slots=True)
class OpenAICompatibleConfig:
    model: str
    api_key: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    organization: str | None = None
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str = "Answer the question using the provided context."
```

Field semantics:

- `model` is a required explicit provider/model identifier
- `base_url` is an explicit endpoint parameter and may be either:
  - a provider base URL such as `https://api.openai.com/v1`
  - or a full `/chat/completions` endpoint URL
- `api_key` may be passed explicitly
- `api_key_env_var` is only a fallback when `api_key` is not provided
- `organization`, `timeout_seconds`, `temperature`, `max_tokens`, and
  `system_prompt` are optional request-shaping parameters

Environment-variable note:

- example-level variables such as `LABELRAG_LLM_MODEL` are convenience settings
  used by runnable example scripts
- they are not part of the core library API contract

### `OpenAICompatibleAnswerGenerator`

```python
class OpenAICompatibleAnswerGenerator:
    def __init__(self, config: OpenAICompatibleConfig) -> None: ...
    def generate(self, question: str, context: str) -> GeneratedAnswer: ...
```

Behavior:

- the adapter targets a minimal non-streaming OpenAI-compatible
  chat-completions API surface
- provider configuration can be supplied entirely through explicit arguments
- environment variables are optional and mainly useful for API-key lookup
- generation metadata should expose the resolved model name and any available
  token-usage metadata

## Generator Protocol

`labelrag` should define an answer-generation abstraction rather than bind
directly to one provider SDK.

```python
class AnswerGenerator(Protocol):
    def generate(self, question: str, context: str) -> GeneratedAnswer: ...
```

Behavior:

- the protocol is synchronous for the first milestone
- generator implementation is injected from outside the retrieval core
- generator metadata should be folded into `RAGAnswerResult.metadata`

## Retrieval Semantics

Baseline retrieval strategy:

- greedy coverage over query label IDs

Tie-break order:

1. larger overlap with remaining query labels
2. larger overlap on query concept IDs when available
3. larger total paragraph label count
4. lexicographically smaller `paragraph_id`

Termination conditions:

- no remaining query labels
- `max_paragraphs` reached
- no paragraph yields positive gain

These retrieval rules are observable behavior and should be tested as part of
the public contract.

## Prompt Rendering

Default prompt context format:

```text
[Paragraph 1 | id=<paragraph_id>]
<paragraph text>

[Paragraph 2 | id=<paragraph_id>]
<paragraph text>
```

If `PromptConfig.include_label_annotations` is enabled, label annotations may be
added, but paragraph text should remain the primary content.

## Persistence Contract

`save(path)` should produce a human-inspectable directory layout similar to:

```text
pipeline_dir/
  manifest.json
  config.json
  label_generator.json
  corpus_index.json
  fit_result.json
```

Persistence responsibilities:

- `manifest.json` stores lightweight snapshot metadata:
  - `labelrag_version`
  - `persistence_format`
  - `artifacts`
- `label_generator.json` stores the fitted `labelgen.LabelGenerator`
- `config.json` stores `RAGPipelineConfig`
- `corpus_index.json` stores `labelrag` retrieval artifacts
- `fit_result.json` stores the fitted `LabelGenerationResult` snapshot exposed
  through `RAGPipeline.fit_result`

Public guarantee:

- a saved and reloaded pipeline should preserve retrieval behavior for the same
  fitted state, question, and config

Current persistence formats:

- `json`
- `json.gz`

Format behavior:

- when `format` is omitted, the implementation auto-detects an existing format
  layout
- when `format` is explicit, it overrides auto-detection
- the current release supports whole-snapshot compression or no compression,
  but not mixed layouts
- snapshots written by `0.0.2` and later are expected to include a manifest
- manifests must include a non-empty `labelrag_version`
- `save(...)` fails explicitly if the current package version cannot be
  determined for manifest writing
- loading pre-`0.0.2` snapshots is a best-effort compatibility path rather than
  a full historical migration guarantee
- for legacy snapshots, missing derived concept inspection tables may be
  rebuilt during load when the necessary paragraph-side concept data is present

## Convenience Re-Exports

For user ergonomics, `labelrag` may re-export the following `labelgen` public
types:

- `Paragraph`
- `LabelGeneratorConfig`

This is optional, but if re-exported it should be documented clearly and remain
stable once published.
