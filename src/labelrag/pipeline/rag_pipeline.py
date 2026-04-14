"""Main RAG pipeline entrypoint."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from labelgen import LabelGenerationResult, LabelGenerator, Paragraph, dump_result, load_result

from labelrag.config import EmbeddingConfig, RAGPipelineConfig
from labelrag.embedding import (
    EmbeddingProvider,
    ParagraphEmbeddingStore,
    SentenceTransformerEmbeddingProvider,
    load_paragraph_embedding_store,
)
from labelrag.generation.generator import AnswerGenerator, GeneratedAnswer
from labelrag.generation.prompt_builder import build_prompt_context
from labelrag.indexing.corpus_index import CorpusIndex, build_corpus_index
from labelrag.io.serialize import (
    backup_other_persistence_format,
    cleanup_persistence_backups,
    corpus_index_from_dict,
    corpus_index_to_dict,
    dump_json,
    ensure_persistence_artifacts_exist,
    has_manifest,
    load_json,
    load_with_optional_gzip,
    manifest_to_dict,
    persistence_path,
    pipeline_config_from_dict,
    pipeline_config_to_dict,
    remove_other_persistence_format,
    resolve_persistence_format,
    restore_persistence_backups,
    save_with_optional_gzip,
    validate_manifest,
)
from labelrag.retrieval.selector import (
    select_concept_overlap_fallback,
    select_greedy_paragraphs,
)
from labelrag.types import (
    ConceptRecord,
    IndexedParagraph,
    LabelRecord,
    QueryAnalysis,
    RAGAnswerResult,
    RetrievalResult,
    RetrievedParagraph,
)


class RAGPipeline:
    """Public pipeline entrypoint for `labelrag`."""

    def __init__(
        self,
        config: RAGPipelineConfig | None = None,
        generator: AnswerGenerator | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or RAGPipelineConfig()
        self.generator = generator
        self._embedding_provider = embedding_provider or _build_embedding_provider(
            self.config.embedding
        )
        self._label_generator = LabelGenerator(self.config.labelgen)
        self._fit_result: LabelGenerationResult | None = None
        self._corpus_index: CorpusIndex | None = None
        self._paragraph_embeddings: ParagraphEmbeddingStore | None = None

    @property
    def fit_result(self) -> LabelGenerationResult | None:
        """Return the most recent fit result when available."""

        return self._fit_result

    @property
    def corpus_index(self) -> CorpusIndex | None:
        """Return the paragraph-side corpus index when available."""

        return self._corpus_index

    def fit(self, paragraphs: list[str] | list[Paragraph]) -> RAGPipeline:
        """Fit the pipeline on a paragraph corpus."""

        if self._embedding_provider is None:
            raise RuntimeError("RAGPipeline.fit() requires an embedding provider.")

        result = self._label_generator.fit_transform(paragraphs)
        self._fit_result = result
        self._corpus_index = build_corpus_index(result)
        paragraph_ids = sorted(self._corpus_index.paragraphs_by_id)
        embeddings = self._embedding_provider.embed_documents(
            [
                self._corpus_index.paragraphs_by_id[paragraph_id].text
                for paragraph_id in paragraph_ids
            ]
        )
        if len(embeddings) != len(paragraph_ids):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of document embeddings."
            )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise RuntimeError("Document embeddings must form a two-dimensional matrix.")
        if self.config.embedding.normalize:
            matrix = _normalize_embedding_rows(matrix)
        self._paragraph_embeddings = ParagraphEmbeddingStore(
            paragraph_ids=paragraph_ids,
            matrix=matrix,
            provider_name=self._embedding_provider.provider_name,
            model_name=self._embedding_provider.model_name,
            normalized=self.config.embedding.normalize,
        )
        return self

    def get_paragraph(self, paragraph_id: str) -> IndexedParagraph | None:
        """Return one indexed paragraph by ID when it exists."""

        self._require_fitted()
        assert self._corpus_index is not None
        return self._corpus_index.paragraphs_by_id.get(paragraph_id)

    def get_label_paragraph_ids(self, label_id: str) -> list[str]:
        """Return paragraph IDs associated with one label."""

        self._require_fitted()
        assert self._corpus_index is not None
        return list(self._corpus_index.paragraph_ids_by_label.get(label_id, []))

    def get_label_paragraphs(self, label_id: str) -> list[IndexedParagraph]:
        """Return paragraph records associated with one label."""

        self._require_fitted()
        assert self._corpus_index is not None
        return [
            self._corpus_index.paragraphs_by_id[paragraph_id]
            for paragraph_id in self.get_label_paragraph_ids(label_id)
            if paragraph_id in self._corpus_index.paragraphs_by_id
        ]

    def get_label(self, label_id: str) -> LabelRecord | None:
        """Return one label inspection record by ID when it exists."""

        self._require_fitted()
        assert self._corpus_index is not None
        if label_id not in self._corpus_index.label_display_names_by_id:
            return None
        return LabelRecord(
            label_id=label_id,
            display_name=self._corpus_index.label_display_names_by_id[label_id],
            concept_ids=list(self._corpus_index.label_concept_ids_by_id.get(label_id, [])),
            paragraph_ids=list(self._corpus_index.paragraph_ids_by_label.get(label_id, [])),
        )

    def get_paragraph_label_ids(self, paragraph_id: str) -> list[str]:
        """Return label IDs associated with one paragraph."""

        self._require_fitted()
        assert self._corpus_index is not None
        return list(self._corpus_index.label_ids_by_paragraph.get(paragraph_id, []))

    def get_paragraph_concept_ids(self, paragraph_id: str) -> list[str]:
        """Return concept IDs associated with one paragraph."""

        self._require_fitted()
        assert self._corpus_index is not None
        return list(self._corpus_index.concept_ids_by_paragraph.get(paragraph_id, []))

    def get_paragraph_labels(self, paragraph_id: str) -> list[LabelRecord]:
        """Return label inspection records associated with one paragraph."""

        self._require_fitted()
        return [
            label
            for label_id in self.get_paragraph_label_ids(paragraph_id)
            if (label := self.get_label(label_id)) is not None
        ]

    def get_paragraph_concepts(self, paragraph_id: str) -> list[ConceptRecord]:
        """Return concept inspection records associated with one paragraph."""

        self._require_fitted()
        return [
            concept
            for concept_id in self.get_paragraph_concept_ids(paragraph_id)
            if (concept := self._get_concept_record(concept_id)) is not None
        ]

    def get_concept_paragraph_ids(self, concept_id: str) -> list[str]:
        """Return paragraph IDs associated with one concept."""

        self._require_fitted()
        assert self._corpus_index is not None
        return list(self._corpus_index.paragraph_ids_by_concept.get(concept_id, []))

    def get_concept_paragraphs(self, concept_id: str) -> list[IndexedParagraph]:
        """Return paragraph records associated with one concept."""

        self._require_fitted()
        assert self._corpus_index is not None
        return [
            self._corpus_index.paragraphs_by_id[paragraph_id]
            for paragraph_id in self.get_concept_paragraph_ids(concept_id)
            if paragraph_id in self._corpus_index.paragraphs_by_id
        ]

    def analyze_query(self, question: str) -> QueryAnalysis:
        """Analyze a query against the fitted label space."""

        self._require_fitted()

        result = self._label_generator.transform([question])
        concept_ids = sorted({mention.concept_id for mention in result.mentions})
        concepts_by_id = {concept.id: concept.normalized for concept in result.concepts}
        label_display_names_by_id = {
            community.id: community.display_name for community in result.communities
        }
        label_ids = list(result.paragraph_labels[0].label_ids) if result.paragraph_labels else []

        return QueryAnalysis(
            query_text=question,
            concepts=[
                concepts_by_id[concept_id]
                for concept_id in concept_ids
                if concept_id in concepts_by_id
            ],
            concept_ids=concept_ids,
            label_ids=label_ids,
            label_display_names=[
                label_display_names_by_id[label_id]
                for label_id in label_ids
                if label_id in label_display_names_by_id
            ],
        )

    def _retrieve_paragraphs(
        self,
        query_analysis: QueryAnalysis,
    ) -> tuple[list[RetrievedParagraph], bool]:
        """Retrieve paragraphs for a analyzed query using greedy label coverage."""

        self._require_fitted()
        assert self._corpus_index is not None
        assert self._paragraph_embeddings is not None
        if self._embedding_provider is None:
            raise RuntimeError(
                "RAGPipeline requires an embedding provider before query-time operations."
            )

        if not query_analysis.label_ids:
            if not self.config.retrieval.allow_label_free_fallback:
                return [], False
            return (
                select_concept_overlap_fallback(
                    query_analysis,
                    self._corpus_index,
                    max_paragraphs=self.config.retrieval.max_paragraphs,
                ),
                False,
            )

        semantic_similarity_by_paragraph = self._semantic_similarity_lookup(
            question=query_analysis.query_text
        )
        return (
            select_greedy_paragraphs(
                query_analysis,
                self._corpus_index,
                max_paragraphs=self.config.retrieval.max_paragraphs,
                semantic_similarity_for_paragraph=(
                    lambda paragraph_id: semantic_similarity_by_paragraph[paragraph_id]
                ),
            ),
            True,
        )

    def build_context(self, question: str) -> RetrievalResult:
        """Build retrieval context for a question."""

        query_analysis = self.analyze_query(question)
        retrieved_paragraphs, semantic_reranking_used = self._retrieve_paragraphs(
            query_analysis
        )
        attempted_covered_label_ids = sorted(
            {
                label_id
                for paragraph in retrieved_paragraphs
                for label_id in paragraph.matched_label_ids
            }
        )
        attempted_uncovered_label_ids = sorted(
            set(query_analysis.label_ids) - set(attempted_covered_label_ids)
        )
        used_label_free_fallback = (
            not query_analysis.label_ids and self.config.retrieval.allow_label_free_fallback
        )
        full_label_coverage_met = not attempted_uncovered_label_ids
        retrieval_strategy = (
            "concept_overlap_fallback"
            if used_label_free_fallback
            else "greedy_label_coverage_semantic_rerank"
        )

        if self.config.retrieval.require_full_label_coverage and not full_label_coverage_met:
            retrieved_paragraphs = []

        prompt_context = build_prompt_context(retrieved_paragraphs, self.config.prompt)
        covered_label_ids = sorted(
            {
                label_id
                for paragraph in retrieved_paragraphs
                for label_id in paragraph.matched_label_ids
            }
        )
        uncovered_label_ids = sorted(set(query_analysis.label_ids) - set(covered_label_ids))

        return RetrievalResult(
            question=question,
            query_analysis=query_analysis,
            retrieved_paragraphs=retrieved_paragraphs,
            prompt_context=prompt_context,
            metadata={
                "covered_label_ids": covered_label_ids,
                "uncovered_label_ids": uncovered_label_ids,
                "attempted_covered_label_ids": attempted_covered_label_ids,
                "attempted_uncovered_label_ids": attempted_uncovered_label_ids,
                "retrieval_strategy": retrieval_strategy,
                "query_label_ids": list(query_analysis.label_ids),
                "retrieval_limit": self.config.retrieval.max_paragraphs,
                "used_label_free_fallback": used_label_free_fallback,
                "require_full_label_coverage": self.config.retrieval.require_full_label_coverage,
                "full_label_coverage_met": full_label_coverage_met,
                "embedding_provider": (
                    self._paragraph_embeddings.provider_name
                    if self._paragraph_embeddings is not None
                    else ""
                ),
                "embedding_model": (
                    self._paragraph_embeddings.model_name
                    if self._paragraph_embeddings is not None
                    else ""
                ),
                "semantic_reranking_enabled": semantic_reranking_used,
            },
        )

    def answer(self, question: str) -> RAGAnswerResult:
        """Answer a question using retrieval and an optional generator."""

        return self._answer_with_optional_generator(question, self.generator)

    def answer_with_generator(
        self,
        question: str,
        generator: AnswerGenerator,
    ) -> RAGAnswerResult:
        """Answer a question with a per-call generator override."""

        return self._answer_with_optional_generator(question, generator)

    def save(
        self,
        path: str | Path,
        format: Literal["json", "json.gz"] | None = None,
    ) -> None:
        """Persist the pipeline state to disk."""

        self._require_fitted()
        assert self._corpus_index is not None
        assert self._fit_result is not None
        assert self._paragraph_embeddings is not None
        fit_result = self._fit_result

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        persistence_format = resolve_persistence_format(destination, format)
        artifact_paths = [
            persistence_path(destination, stem, persistence_format)
            for stem in ("config", "label_generator", "fit_result", "corpus_index", "manifest")
        ]
        embeddings_path = destination / "paragraph_embeddings.npz"
        backups = backup_other_persistence_format(destination, persistence_format)
        embeddings_backup = embeddings_path.with_name(f"{embeddings_path.name}.bak")
        if embeddings_path.exists():
            embeddings_backup.unlink(missing_ok=True)
            embeddings_path.rename(embeddings_backup)
        try:
            dump_json(
                pipeline_config_to_dict(self.config),
                artifact_paths[0],
            )
            save_with_optional_gzip(
                artifact_paths[1],
                self._label_generator.save,
            )
            save_with_optional_gzip(
                artifact_paths[2],
                lambda artifact_path: dump_result(fit_result, artifact_path),
            )
            dump_json(
                corpus_index_to_dict(self._corpus_index),
                artifact_paths[3],
            )
            self._paragraph_embeddings.save(embeddings_path)
            dump_json(
                manifest_to_dict(
                    labelrag_version=_package_version(),
                    persistence_format=persistence_format,
                    artifacts=[artifact_path.name for artifact_path in artifact_paths]
                    + [embeddings_path.name],
                ),
                artifact_paths[4],
            )
        except Exception:
            for artifact_path in artifact_paths:
                artifact_path.unlink(missing_ok=True)
            embeddings_path.unlink(missing_ok=True)
            if embeddings_backup.exists():
                embeddings_backup.rename(embeddings_path)
            restore_persistence_backups(backups)
            raise

        embeddings_backup.unlink(missing_ok=True)
        cleanup_persistence_backups(backups)
        remove_other_persistence_format(destination, persistence_format)

    @classmethod
    def load(
        cls,
        path: str | Path,
        format: Literal["json", "json.gz"] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> RAGPipeline:
        """Load a pipeline from disk."""

        source = Path(path)
        persistence_format = resolve_persistence_format(source, format)
        include_manifest = has_manifest(source, persistence_format)
        ensure_persistence_artifacts_exist(
            source,
            persistence_format,
            include_manifest=include_manifest,
            include_embedding_artifact=False,
        )
        manifest_data: dict[str, Any] | None = None
        if include_manifest:
            manifest_data = load_json(persistence_path(source, "manifest", persistence_format))
            validate_manifest(
                manifest_data,
                format=persistence_format,
            )
        config = pipeline_config_from_dict(
            load_json(persistence_path(source, "config", persistence_format))
        )
        pipeline = cls(config=config, embedding_provider=embedding_provider)
        pipeline._label_generator = load_with_optional_gzip(
            persistence_path(source, "label_generator", persistence_format),
            LabelGenerator.load,
        )
        pipeline._fit_result = load_with_optional_gzip(
            persistence_path(source, "fit_result", persistence_format),
            load_result,
        )
        pipeline._corpus_index = corpus_index_from_dict(
            load_json(persistence_path(source, "corpus_index", persistence_format)),
            pipeline._fit_result,
        )
        embeddings_path = source / "paragraph_embeddings.npz"
        embedding_artifact_expected = manifest_data is not None and "paragraph_embeddings.npz" in (
            [str(value) for value in manifest_data.get("artifacts", [])]
        )
        if embeddings_path.is_file():
            pipeline._paragraph_embeddings = load_paragraph_embedding_store(embeddings_path)
        elif embedding_artifact_expected:
            raise RuntimeError("Missing persistence artifact: paragraph_embeddings.npz.")
        else:
            pipeline._paragraph_embeddings = _rebuild_legacy_paragraph_embeddings(
                pipeline._corpus_index,
                pipeline._embedding_provider,
                pipeline.config.embedding.normalize,
            )
        _validate_paragraph_embeddings(pipeline._paragraph_embeddings, pipeline._corpus_index)
        return pipeline

    def _require_fitted(self) -> None:
        """Validate that the pipeline has already been fitted."""

        if self._corpus_index is None:
            raise RuntimeError("RAGPipeline requires fit() before query-time operations.")

    def _answer_with_optional_generator(
        self,
        question: str,
        generator: AnswerGenerator | None,
    ) -> RAGAnswerResult:
        """Build an answer result with an optional injected generator."""

        retrieval_result = self.build_context(question)
        generated_answer = self._generate_answer(
            question,
            retrieval_result.prompt_context,
            generator,
        )

        return RAGAnswerResult(
            question=question,
            answer_text=generated_answer.text,
            query_analysis=retrieval_result.query_analysis,
            retrieved_paragraphs=retrieval_result.retrieved_paragraphs,
            prompt_context=retrieval_result.prompt_context,
            metadata={
                **retrieval_result.metadata,
                "generator_name": _generator_name(generator),
                "generation_model": _generation_model(generated_answer),
                "generation_metadata": dict(generated_answer.metadata),
            },
        )

    def _generate_answer(
        self,
        question: str,
        context: str,
        generator: AnswerGenerator | None,
    ) -> GeneratedAnswer:
        """Generate an answer or return an empty placeholder when no generator is configured."""

        if generator is None:
            return GeneratedAnswer(text="", metadata={})
        return generator.generate(question, context)

    def _get_concept_record(self, concept_id: str) -> ConceptRecord | None:
        """Return one concept inspection record by ID when it exists."""

        assert self._corpus_index is not None
        if concept_id not in self._corpus_index.concept_texts_by_id:
            return None
        return ConceptRecord(
            concept_id=concept_id,
            text=self._corpus_index.concept_texts_by_id[concept_id],
            paragraph_ids=list(self._corpus_index.paragraph_ids_by_concept.get(concept_id, [])),
        )

    def _semantic_similarity_lookup(self, *, question: str) -> dict[str, float]:
        """Build paragraph-ID similarity scores for one query."""

        assert self._paragraph_embeddings is not None
        assert self._embedding_provider is not None
        query_embedding = np.asarray(
            self._embedding_provider.embed_query(question),
            dtype=np.float32,
        )
        if query_embedding.ndim != 1:
            raise RuntimeError("Query embedding must be one-dimensional.")
        if self._paragraph_embeddings.normalized:
            query_embedding = _normalize_embedding(query_embedding)
        matrix = self._paragraph_embeddings.matrix
        if matrix.shape[1] != query_embedding.shape[0]:
            raise RuntimeError(
                "Query embedding dimensionality does not match stored paragraph embeddings."
            )
        similarities = matrix @ query_embedding
        return {
            paragraph_id: float(similarity)
            for paragraph_id, similarity in zip(
                self._paragraph_embeddings.paragraph_ids,
                similarities.tolist(),
                strict=True,
            )
        }


def _generator_name(generator: AnswerGenerator | None) -> str:
    """Return a stable generator identifier for metadata."""

    if generator is None:
        return ""
    return type(generator).__name__


def _generation_model(answer: GeneratedAnswer) -> str:
    """Return a best-effort generation model name from generator metadata."""

    model = answer.metadata.get("model")
    if not isinstance(model, str):
        return ""
    return model


def _package_version() -> str:
    """Return the installed package version when available."""

    try:
        return package_version("labelrag")
    except PackageNotFoundError as err:
        pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject_path.is_file():
            with pyproject_path.open("rb") as handle:
                project_table = tomllib.load(handle).get("project", {})
            version = project_table.get("version")
            if isinstance(version, str) and version:
                return version
        raise RuntimeError(
            "Unable to determine labelrag version for persistence manifest."
        ) from err


def _build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider | None:
    """Build a supported embedding provider from configuration."""

    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(config)
    return None


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize one embedding vector for cosine-style similarity."""

    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise RuntimeError("Embedding provider returned a zero-norm embedding.")
    return embedding / norm


def _normalize_embedding_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize embedding rows for cosine-style similarity."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise RuntimeError("Embedding provider returned a zero-norm document embedding.")
    return matrix / norms


def _validate_paragraph_embeddings(
    store: ParagraphEmbeddingStore,
    corpus_index: CorpusIndex,
) -> None:
    """Validate that stored embeddings align with the fitted paragraph index."""

    paragraph_ids = sorted(corpus_index.paragraphs_by_id)
    if store.paragraph_ids != paragraph_ids:
        raise RuntimeError(
            "Stored paragraph embeddings do not align with the fitted paragraph IDs."
        )
    if store.matrix.ndim != 2:
        raise RuntimeError("Stored paragraph embeddings must be a two-dimensional matrix.")
    if store.matrix.shape[0] != len(store.paragraph_ids):
        raise RuntimeError(
            "Stored paragraph embeddings row count does not match stored paragraph IDs."
        )


def _rebuild_legacy_paragraph_embeddings(
    corpus_index: CorpusIndex | None,
    embedding_provider: EmbeddingProvider | None,
    normalize: bool,
) -> ParagraphEmbeddingStore:
    """Rebuild paragraph embeddings for snapshots that predate the embedding artifact."""

    if corpus_index is None:
        raise RuntimeError("Cannot rebuild legacy paragraph embeddings without a corpus index.")
    if embedding_provider is None:
        raise RuntimeError(
            "Legacy snapshots without paragraph_embeddings.npz require an embedding provider "
            "to rebuild paragraph embeddings during load."
        )

    paragraph_ids = sorted(corpus_index.paragraphs_by_id)
    embeddings = embedding_provider.embed_documents(
        [corpus_index.paragraphs_by_id[paragraph_id].text for paragraph_id in paragraph_ids]
    )
    if len(embeddings) != len(paragraph_ids):
        raise RuntimeError(
            "Embedding provider returned an unexpected number of document embeddings while "
            "rebuilding a legacy snapshot."
        )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("Document embeddings must form a two-dimensional matrix.")
    if normalize:
        matrix = _normalize_embedding_rows(matrix)
    return ParagraphEmbeddingStore(
        paragraph_ids=paragraph_ids,
        matrix=matrix,
        provider_name=embedding_provider.provider_name,
        model_name=embedding_provider.model_name,
        normalized=normalize,
    )
