"""Main RAG pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from labelgen import LabelGenerationResult, LabelGenerator, Paragraph, dump_result, load_result

from labelrag.config import RAGPipelineConfig
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
    load_json,
    load_with_optional_gzip,
    persistence_path,
    pipeline_config_from_dict,
    pipeline_config_to_dict,
    remove_other_persistence_format,
    resolve_persistence_format,
    restore_persistence_backups,
    save_with_optional_gzip,
)
from labelrag.retrieval.selector import (
    select_concept_overlap_fallback,
    select_greedy_paragraphs,
)
from labelrag.types import (
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
    ) -> None:
        self.config = config or RAGPipelineConfig()
        self.generator = generator
        self._label_generator = LabelGenerator(self.config.labelgen)
        self._fit_result: LabelGenerationResult | None = None
        self._corpus_index: CorpusIndex | None = None

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

        result = self._label_generator.fit_transform(paragraphs)
        self._fit_result = result
        self._corpus_index = build_corpus_index(result)
        return self

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

    def _retrieve_paragraphs(self, query_analysis: QueryAnalysis) -> list[RetrievedParagraph]:
        """Retrieve paragraphs for a analyzed query using greedy label coverage."""

        self._require_fitted()
        assert self._corpus_index is not None

        if not query_analysis.label_ids:
            if not self.config.retrieval.allow_label_free_fallback:
                return []
            return select_concept_overlap_fallback(
                query_analysis,
                self._corpus_index,
                max_paragraphs=self.config.retrieval.max_paragraphs,
            )

        return select_greedy_paragraphs(
            query_analysis,
            self._corpus_index,
            max_paragraphs=self.config.retrieval.max_paragraphs,
        )

    def build_context(self, question: str) -> RetrievalResult:
        """Build retrieval context for a question."""

        query_analysis = self.analyze_query(question)
        retrieved_paragraphs = self._retrieve_paragraphs(query_analysis)
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
            else "greedy_label_coverage"
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
        fit_result = self._fit_result

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        persistence_format = resolve_persistence_format(destination, format)
        artifact_paths = [
            persistence_path(destination, stem, persistence_format)
            for stem in ("config", "label_generator", "fit_result", "corpus_index")
        ]
        backups = backup_other_persistence_format(destination, persistence_format)
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
        except Exception:
            for artifact_path in artifact_paths:
                artifact_path.unlink(missing_ok=True)
            restore_persistence_backups(backups)
            raise

        cleanup_persistence_backups(backups)
        remove_other_persistence_format(destination, persistence_format)

    @classmethod
    def load(
        cls,
        path: str | Path,
        format: Literal["json", "json.gz"] | None = None,
    ) -> RAGPipeline:
        """Load a pipeline from disk."""

        source = Path(path)
        persistence_format = resolve_persistence_format(source, format)
        ensure_persistence_artifacts_exist(source, persistence_format)
        config = pipeline_config_from_dict(
            load_json(persistence_path(source, "config", persistence_format))
        )
        pipeline = cls(config=config)
        pipeline._label_generator = load_with_optional_gzip(
            persistence_path(source, "label_generator", persistence_format),
            LabelGenerator.load,
        )
        pipeline._fit_result = load_with_optional_gzip(
            persistence_path(source, "fit_result", persistence_format),
            load_result,
        )
        pipeline._corpus_index = corpus_index_from_dict(
            load_json(persistence_path(source, "corpus_index", persistence_format))
        )
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
