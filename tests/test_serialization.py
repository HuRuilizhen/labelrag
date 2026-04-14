"""Tests for pipeline save/load behavior."""

import json
from dataclasses import dataclass
from io import BufferedReader
from pathlib import Path

import numpy as np
import pytest

from labelrag import EmbeddingConfig, RAGPipeline, RAGPipelineConfig
from labelrag.generation.generator import GeneratedAnswer
from labelrag.io.serialize import pipeline_config_from_dict
from labelrag.pipeline import rag_pipeline as rag_pipeline_module
from support import AlternateStubEmbeddingProvider, StubEmbeddingProvider


@dataclass
class StubGenerator:
    """Simple generator stub for load-time answer integration tests."""

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Return a deterministic generated answer."""

        return GeneratedAnswer(
            text=f"answer for: {question}",
            metadata={"model": "stub-model", "has_context": bool(context)},
        )


def _build_pipeline() -> RAGPipeline:
    """Construct a pipeline with a deterministic embedding provider."""

    return RAGPipeline(RAGPipelineConfig(), embedding_provider=StubEmbeddingProvider())


def test_save_and_load_preserve_build_context_behavior(tmp_path: Path) -> None:
    """A saved and loaded pipeline should preserve retrieval behavior."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    original = pipeline.build_context("How do developers use language models?")
    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    restored = loaded.build_context("How do developers use language models?")

    assert loaded.fit_result == pipeline.fit_result
    assert restored.query_analysis == original.query_analysis
    assert restored.retrieved_paragraphs == original.retrieved_paragraphs
    assert restored.prompt_context == original.prompt_context
    assert restored.metadata == original.metadata


def test_save_writes_expected_files(tmp_path: Path) -> None:
    """Saving should produce the expected human-inspectable file layout."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    assert (output_dir / "config.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "label_generator.json").is_file()
    assert (output_dir / "fit_result.json").is_file()
    assert (output_dir / "corpus_index.json").is_file()
    assert (output_dir / "paragraph_embeddings.npz").is_file()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest["labelrag_version"], str)
    assert manifest["labelrag_version"]
    assert manifest["persistence_format"] == "json"
    assert manifest["artifacts"] == [
        "config.json",
        "label_generator.json",
        "fit_result.json",
        "corpus_index.json",
        "manifest.json",
        "paragraph_embeddings.npz",
    ]


def test_save_and_load_support_json_gz_round_trip(tmp_path: Path) -> None:
    """Compressed persistence should preserve retrieval behavior."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    original = pipeline.build_context("How do developers use language models?")
    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")

    assert (output_dir / "config.json.gz").is_file()
    assert (output_dir / "manifest.json.gz").is_file()
    assert (output_dir / "label_generator.json.gz").is_file()
    assert (output_dir / "fit_result.json.gz").is_file()
    assert (output_dir / "corpus_index.json.gz").is_file()
    assert (output_dir / "paragraph_embeddings.npz").is_file()

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    restored = loaded.build_context("How do developers use language models?")

    assert restored.query_analysis == original.query_analysis
    assert restored.retrieved_paragraphs == original.retrieved_paragraphs
    assert restored.prompt_context == original.prompt_context
    assert restored.metadata == original.metadata


def test_save_reuses_detected_existing_persistence_format(tmp_path: Path) -> None:
    """Saving without an explicit format should preserve an existing compressed layout."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")
    pipeline.save(output_dir)

    assert (output_dir / "manifest.json.gz").is_file()
    assert (output_dir / "config.json.gz").is_file()
    assert not (output_dir / "config.json").exists()


def test_load_accepts_explicit_persistence_format_override(tmp_path: Path) -> None:
    """Loading should support an explicit format override."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")

    loaded = RAGPipeline.load(
        output_dir,
        format="json.gz",
        embedding_provider=StubEmbeddingProvider(),
    )
    result = loaded.build_context("How do developers use language models?")

    assert result.prompt_context
    assert result.metadata["retrieval_strategy"] == "greedy_label_coverage_semantic_rerank"


def test_json_and_json_gz_round_trips_are_behaviorally_equivalent(tmp_path: Path) -> None:
    """Compressed and uncompressed persistence should restore the same behavior."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    json_dir = tmp_path / "pipeline-json"
    gzip_dir = tmp_path / "pipeline-json-gz"
    pipeline.save(json_dir)
    pipeline.save(gzip_dir, format="json.gz")

    json_loaded = RAGPipeline.load(json_dir, embedding_provider=StubEmbeddingProvider())
    gzip_loaded = RAGPipeline.load(gzip_dir, embedding_provider=StubEmbeddingProvider())
    question = "How do developers use language models?"

    assert json_loaded.build_context(question) == gzip_loaded.build_context(question)


def test_load_rejects_incorrect_explicit_persistence_format(tmp_path: Path) -> None:
    """Loading with an explicit mismatched format should fail clearly."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")

    with pytest.raises(RuntimeError, match="Missing persistence artifacts for format `json`"):
        RAGPipeline.load(output_dir, format="json")


def test_failed_format_migration_restores_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed save should restore the previous persistence snapshot."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir, format="json.gz")

    original_dump_result = rag_pipeline_module.dump_result

    def fail_dump_result(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(rag_pipeline_module, "dump_result", fail_dump_result)

    with pytest.raises(RuntimeError, match="simulated failure"):
        pipeline.save(output_dir, format="json")

    monkeypatch.setattr(rag_pipeline_module, "dump_result", original_dump_result)

    assert (output_dir / "config.json.gz").is_file()
    assert (output_dir / "manifest.json.gz").is_file()
    assert (output_dir / "label_generator.json.gz").is_file()
    assert (output_dir / "fit_result.json.gz").is_file()
    assert (output_dir / "corpus_index.json.gz").is_file()
    assert (output_dir / "paragraph_embeddings.npz").is_file()
    assert not (output_dir / "config.json").exists()

    loaded = RAGPipeline.load(
        output_dir,
        format="json.gz",
        embedding_provider=StubEmbeddingProvider(),
    )
    result = loaded.build_context("How do developers use language models?")
    assert result.prompt_context


def test_load_supports_legacy_snapshot_without_manifest(tmp_path: Path) -> None:
    """Loading should remain backward compatible with snapshots that predate manifests."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "manifest.json").unlink()

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    result = loaded.build_context("How do developers use language models?")

    assert result.prompt_context
    assert loaded.fit_result == pipeline.fit_result


def test_load_supports_legacy_manifest_snapshot_without_embedding_artifact(
    tmp_path: Path,
) -> None:
    """Legacy snapshots with manifests should still load without embedding artifacts."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "paragraph_embeddings.npz").unlink()

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labelrag_version"] = "0.0.3"
    manifest["artifacts"] = [
        artifact_name
        for artifact_name in manifest["artifacts"]
        if artifact_name != "paragraph_embeddings.npz"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config_path = output_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data.pop("embedding", None)
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    result = loaded.build_context("How do developers use language models?")

    assert result.prompt_context
    assert loaded.fit_result == pipeline.fit_result


def test_pipeline_config_from_dict_defaults_missing_embedding_config() -> None:
    """Legacy configs without an embedding block should use the default embedding config."""

    config = pipeline_config_from_dict(
        {
            "labelgen": {},
            "retrieval": {},
            "prompt": {},
        }
    )

    assert config.embedding == EmbeddingConfig()


def test_load_rebuilds_legacy_concept_reverse_lookups(tmp_path: Path) -> None:
    """Legacy snapshots should rebuild derived concept inspection state on load."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )
    assert pipeline.corpus_index is not None

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    corpus_index_path = output_dir / "corpus_index.json"
    corpus_index_data = json.loads(corpus_index_path.read_text(encoding="utf-8"))
    corpus_index_data.pop("paragraph_ids_by_concept", None)
    corpus_index_data.pop("concept_texts_by_id", None)
    corpus_index_path.write_text(json.dumps(corpus_index_data, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").unlink()

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    paragraph_id = sorted(pipeline.corpus_index.paragraphs_by_id)[0]
    paragraph = pipeline.get_paragraph(paragraph_id)
    assert paragraph is not None

    concept_id = paragraph.concept_ids[0]
    assert loaded.get_concept_paragraph_ids(concept_id) == pipeline.get_concept_paragraph_ids(
        concept_id
    )
    assert loaded.corpus_index is not None
    assert loaded.corpus_index.concept_texts_by_id[concept_id] in paragraph.concept_texts


def test_load_rebuilds_legacy_embeddings_without_embedding_artifact(tmp_path: Path) -> None:
    """Legacy snapshots should rebuild paragraph embeddings when the artifact is absent."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "manifest.json").unlink()
    (output_dir / "paragraph_embeddings.npz").unlink()

    config_path = output_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data.pop("embedding", None)
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    result = loaded.build_context("How do developers use language models?")

    assert result.prompt_context
    assert loaded.corpus_index is not None
    assert loaded.fit_result == pipeline.fit_result


def test_load_rejects_legacy_snapshot_without_available_embedding_provider(
    tmp_path: Path,
) -> None:
    """Legacy snapshots without embedding artifacts should require an available provider."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "manifest.json").unlink()
    (output_dir / "paragraph_embeddings.npz").unlink()
    config_path = output_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data["embedding"]["provider"] = "unsupported-provider"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unsupported embedding provider"):
        RAGPipeline.load(output_dir, embedding_provider=None)


def test_load_rebuilds_legacy_embeddings_via_default_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy snapshots should rebuild embeddings through config-driven provider construction."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "manifest.json").unlink()
    (output_dir / "paragraph_embeddings.npz").unlink()

    def build_stub_provider(_: object) -> StubEmbeddingProvider:
        return StubEmbeddingProvider()

    monkeypatch.setattr(
        rag_pipeline_module,
        "SentenceTransformerEmbeddingProvider",
        build_stub_provider,
    )

    loaded = RAGPipeline.load(output_dir, embedding_provider=None)

    assert loaded.build_context("How do developers use language models?").prompt_context


def test_load_rejects_manifest_without_labelrag_version(tmp_path: Path) -> None:
    """Loading should fail when the manifest omits the required package version field."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("labelrag_version", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="labelrag_version"):
        RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())


def test_save_uses_pyproject_version_when_package_metadata_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving should still write a non-empty version when package metadata is unavailable."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    def raise_package_not_found(_: str) -> str:
        raise rag_pipeline_module.PackageNotFoundError

    monkeypatch.setattr(rag_pipeline_module, "package_version", raise_package_not_found)

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    with Path("pyproject.toml").open("rb") as handle:
        expected_version = rag_pipeline_module.tomllib.load(handle)["project"]["version"]
    assert manifest["labelrag_version"] == expected_version


def test_save_fails_when_no_manifest_version_source_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving should fail clearly when no manifest version source can be resolved."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    def raise_package_not_found(_: str) -> str:
        raise rag_pipeline_module.PackageNotFoundError

    def load_project_without_version(_: BufferedReader) -> dict[str, dict[str, object]]:
        return {"project": {}}

    monkeypatch.setattr(rag_pipeline_module, "package_version", raise_package_not_found)
    monkeypatch.setattr(rag_pipeline_module.tomllib, "load", load_project_without_version)

    output_dir = tmp_path / "pipeline"
    with pytest.raises(RuntimeError, match="Unable to determine labelrag version"):
        pipeline.save(output_dir)


def test_load_rebuilds_legacy_label_concept_ids(tmp_path: Path) -> None:
    """Legacy snapshots should rebuild label concept IDs for record-oriented inspection."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )
    assert pipeline.corpus_index is not None

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    corpus_index_path = output_dir / "corpus_index.json"
    corpus_index_data = json.loads(corpus_index_path.read_text(encoding="utf-8"))
    corpus_index_data.pop("label_concept_ids_by_id", None)
    corpus_index_path.write_text(json.dumps(corpus_index_data, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").unlink()

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    paragraph_id = sorted(pipeline.corpus_index.paragraphs_by_id)[0]
    paragraph = pipeline.get_paragraph(paragraph_id)
    assert paragraph is not None
    label_id = paragraph.label_ids[0]

    assert loaded.get_label(label_id) == pipeline.get_label(label_id)


def test_load_supports_answer_with_generator(tmp_path: Path) -> None:
    """A loaded pipeline should still support end-to-end answer generation."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    loaded = RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())
    result = loaded.answer_with_generator(
        "How do developers use language models?",
        StubGenerator(),
    )

    assert result.answer_text == "answer for: How do developers use language models?"
    assert result.metadata["generator_name"] == "StubGenerator"
    assert result.metadata["generation_model"] == "stub-model"


def test_load_rejects_missing_embedding_artifact(tmp_path: Path) -> None:
    """Loading should fail clearly when the embedding artifact is missing."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    (output_dir / "paragraph_embeddings.npz").unlink()

    with pytest.raises(RuntimeError, match="paragraph_embeddings.npz"):
        RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())


def test_load_rejects_inconsistent_embedding_artifact(tmp_path: Path) -> None:
    """Loading should fail when stored paragraph embeddings do not align with paragraph IDs."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)
    np.savez(
        output_dir / "paragraph_embeddings.npz",
        paragraph_ids=np.asarray(["wrong-paragraph-id"], dtype=str),
        matrix=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        provider_name=np.asarray("stub"),
        model_name=np.asarray("stub-embedding-model"),
        normalized=np.asarray(True),
    )

    with pytest.raises(RuntimeError, match="paragraph IDs"):
        RAGPipeline.load(output_dir, embedding_provider=StubEmbeddingProvider())


def test_load_rejects_embedding_provider_mismatch(tmp_path: Path) -> None:
    """Loading should fail when the active provider does not match persisted embeddings."""

    pipeline = _build_pipeline()
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    with pytest.raises(RuntimeError, match="different embedding provider"):
        RAGPipeline.load(output_dir, embedding_provider=AlternateStubEmbeddingProvider())
