"""Generic JSON helpers for `labelrag`."""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeVar, cast

from labelgen import LabelGenerationResult
from labelgen.io.serialize import config_from_dict, config_to_dict

from labelrag.config import PromptConfig, RAGPipelineConfig, RetrievalConfig
from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.types import IndexedParagraph

PersistenceFormat = Literal["json", "json.gz"]
_ARTIFACT_STEMS = ("manifest", "config", "label_generator", "fit_result", "corpus_index")
_CORE_ARTIFACT_STEMS = ("config", "label_generator", "fit_result", "corpus_index")
_T = TypeVar("_T")


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    """Serialize a JSON object to disk."""

    destination = Path(path)
    dump_text(json.dumps(data, indent=2, sort_keys=True), destination)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    source = Path(path)
    data = json.loads(load_text(source))
    if not isinstance(data, dict):
        raise TypeError("Serialized object must be a JSON object.")
    return _as_string_key_dict(cast(object, data))


def dump_text(data: str, path: str | Path) -> None:
    """Serialize text to disk with optional gzip compression."""

    destination = Path(path)
    if destination.name.endswith(".gz"):
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            handle.write(data)
        return
    destination.write_text(data, encoding="utf-8")


def load_text(path: str | Path) -> str:
    """Load text from disk with optional gzip decompression."""

    source = Path(path)
    if source.name.endswith(".gz"):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return handle.read()
    return source.read_text(encoding="utf-8")


def save_with_optional_gzip(
    path: str | Path,
    writer: Callable[[Path], None],
) -> None:
    """Write an artifact directly or through a temporary gzip wrapper."""

    destination = Path(path)
    if not destination.name.endswith(".gz"):
        writer(destination)
        return

    with TemporaryDirectory() as tmp_dir:
        temporary_path = Path(tmp_dir) / destination.name.removesuffix(".gz")
        writer(temporary_path)
        with (
            temporary_path.open("rb") as source_handle,
            gzip.open(destination, "wb") as dest_handle,
        ):
            shutil.copyfileobj(source_handle, dest_handle)


def load_with_optional_gzip(
    path: str | Path,
    loader: Callable[[Path], _T],
) -> _T:
    """Load an artifact directly or through a temporary gzip wrapper."""

    source = Path(path)
    if not source.name.endswith(".gz"):
        return loader(source)

    with TemporaryDirectory() as tmp_dir:
        temporary_path = Path(tmp_dir) / source.name.removesuffix(".gz")
        with gzip.open(source, "rb") as source_handle, temporary_path.open("wb") as dest_handle:
            shutil.copyfileobj(source_handle, dest_handle)
        return loader(temporary_path)


def persistence_path(root: str | Path, stem: str, format: PersistenceFormat) -> Path:
    """Build a persistence artifact path for a known format."""

    suffix = ".json.gz" if format == "json.gz" else ".json"
    return Path(root) / f"{stem}{suffix}"


def resolve_persistence_format(
    root: str | Path,
    format: PersistenceFormat | None = None,
) -> PersistenceFormat:
    """Resolve a persistence format from explicit input or on-disk artifacts."""

    if format is not None:
        return _normalize_persistence_format(format)

    source = Path(root)
    json_paths = [persistence_path(source, stem, "json") for stem in _CORE_ARTIFACT_STEMS]
    gzip_paths = [persistence_path(source, stem, "json.gz") for stem in _CORE_ARTIFACT_STEMS]

    json_exists = [path.exists() for path in json_paths]
    gzip_exists = [path.exists() for path in gzip_paths]

    if any(json_exists) and any(gzip_exists):
        raise RuntimeError(
            "Detected mixed persistence formats. Pass `format` explicitly or clean the "
            "target directory."
        )
    if any(json_exists):
        if not all(json_exists):
            raise RuntimeError(
                "Detected incomplete JSON persistence artifacts. Pass `format` explicitly "
                "or clean the target directory."
            )
        return "json"
    if any(gzip_exists):
        if not all(gzip_exists):
            raise RuntimeError(
                "Detected incomplete JSON.GZ persistence artifacts. Pass `format` "
                "explicitly or clean the target directory."
            )
        return "json.gz"
    return "json"


def remove_other_persistence_format(root: str | Path, format: PersistenceFormat) -> None:
    """Remove stale persistence files from the non-selected format."""

    other = "json.gz" if format == "json" else "json"
    for stem in _ARTIFACT_STEMS:
        persistence_path(root, stem, other).unlink(missing_ok=True)


def backup_other_persistence_format(
    root: str | Path,
    format: PersistenceFormat,
) -> list[tuple[Path, Path]]:
    """Rename stale artifacts from the non-selected format to backup files."""

    source = Path(root)
    other = "json.gz" if format == "json" else "json"
    backups: list[tuple[Path, Path]] = []
    for stem in _ARTIFACT_STEMS:
        original = persistence_path(source, stem, other)
        if not original.exists():
            continue
        backup = original.with_name(f"{original.name}.bak")
        backup.unlink(missing_ok=True)
        original.rename(backup)
        backups.append((original, backup))
    return backups


def restore_persistence_backups(backups: list[tuple[Path, Path]]) -> None:
    """Restore backed up artifacts after a failed save attempt."""

    for original, backup in backups:
        if original.exists():
            original.unlink()
        if backup.exists():
            backup.rename(original)


def cleanup_persistence_backups(backups: list[tuple[Path, Path]]) -> None:
    """Remove backup files after a successful save attempt."""

    for _, backup in backups:
        backup.unlink(missing_ok=True)


def ensure_persistence_artifacts_exist(
    root: str | Path,
    format: PersistenceFormat,
    *,
    include_manifest: bool = True,
) -> None:
    """Validate that all required artifacts exist for the chosen format."""

    source = Path(root)
    stems = _ARTIFACT_STEMS if include_manifest else _CORE_ARTIFACT_STEMS
    missing_paths = [
        str(path.name)
        for stem in stems
        if not (path := persistence_path(source, stem, format)).is_file()
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise RuntimeError(
            f"Missing persistence artifacts for format `{format}`: {missing}."
        )


def has_manifest(root: str | Path, format: PersistenceFormat) -> bool:
    """Return whether a persistence snapshot includes a manifest artifact."""

    return persistence_path(root, "manifest", format).is_file()


def manifest_to_dict(
    *,
    labelrag_version: str,
    persistence_format: PersistenceFormat,
    artifacts: list[str],
) -> dict[str, Any]:
    """Build a lightweight persistence manifest payload."""

    return {
        "labelrag_version": labelrag_version,
        "persistence_format": persistence_format,
        "artifacts": list(artifacts),
    }


def validate_manifest(
    data: dict[str, Any],
    *,
    format: PersistenceFormat,
) -> None:
    """Validate the minimal shape of a persisted manifest."""

    labelrag_version_value = data.get("labelrag_version")
    if not isinstance(labelrag_version_value, str):
        raise RuntimeError("Persistence manifest must include `labelrag_version`.")
    labelrag_version = labelrag_version_value
    if not labelrag_version:
        raise RuntimeError("Persistence manifest must include a non-empty `labelrag_version`.")

    persisted_format = _as_string(data.get("persistence_format"))
    if persisted_format != format:
        raise RuntimeError(
            "Persistence manifest format does not match the requested snapshot format."
        )

    artifact_names = _as_string_list(data.get("artifacts"))
    expected_artifacts = [
        persistence_path(".", stem, format).name
        for stem in _ARTIFACT_STEMS
    ]
    missing_artifacts = [
        artifact_name
        for artifact_name in expected_artifacts
        if artifact_name not in artifact_names
    ]
    if missing_artifacts:
        missing = ", ".join(missing_artifacts)
        raise RuntimeError(
            "Persistence manifest is missing required artifact entries: "
            f"{missing}."
        )


def _normalize_persistence_format(value: str) -> PersistenceFormat:
    """Validate a persistence format value."""

    if value not in {"json", "json.gz"}:
        raise ValueError("Persistence format must be either `json` or `json.gz`.")
    return cast(PersistenceFormat, value)


def _as_string_key_dict(value: object) -> dict[str, Any]:
    """Normalize a mapping into a string-key dictionary."""

    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")

    normalized: dict[str, Any] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings.")
        normalized[key] = item
    return normalized


def pipeline_config_to_dict(config: RAGPipelineConfig) -> dict[str, Any]:
    """Convert a pipeline config into a JSON-serializable dictionary."""

    return {
        "labelgen": config_to_dict(config.labelgen),
        "retrieval": asdict(config.retrieval),
        "prompt": asdict(config.prompt),
    }


def pipeline_config_from_dict(data: dict[str, Any]) -> RAGPipelineConfig:
    """Reconstruct a pipeline config from serialized data."""

    retrieval_data = _as_string_key_dict(data.get("retrieval"))
    prompt_data = _as_string_key_dict(data.get("prompt"))
    return RAGPipelineConfig(
        labelgen=config_from_dict(_as_string_key_dict(data.get("labelgen"))),
        retrieval=RetrievalConfig(**retrieval_data),
        prompt=PromptConfig(**prompt_data),
    )


def corpus_index_to_dict(index: CorpusIndex) -> dict[str, Any]:
    """Convert a corpus index into a JSON-serializable dictionary."""

    return {
        "paragraphs_by_id": {
            paragraph_id: asdict(paragraph)
            for paragraph_id, paragraph in index.paragraphs_by_id.items()
        },
        "paragraph_ids_by_label": index.paragraph_ids_by_label,
        "label_ids_by_paragraph": index.label_ids_by_paragraph,
        "concept_ids_by_paragraph": index.concept_ids_by_paragraph,
        "paragraph_ids_by_concept": index.paragraph_ids_by_concept,
        "label_display_names_by_id": index.label_display_names_by_id,
        "label_concept_ids_by_id": index.label_concept_ids_by_id,
        "concept_texts_by_id": index.concept_texts_by_id,
    }


def corpus_index_from_dict(
    data: dict[str, Any],
    fit_result: LabelGenerationResult | None = None,
) -> CorpusIndex:
    """Reconstruct a corpus index from serialized data."""

    paragraphs_by_id_data = _as_string_key_dict(data.get("paragraphs_by_id"))
    paragraphs_by_id = {
        paragraph_id: IndexedParagraph(**_as_string_key_dict(paragraph_data))
        for paragraph_id, paragraph_data in paragraphs_by_id_data.items()
    }
    concept_ids_by_paragraph = {
        paragraph_id: _as_string_list(value)
        for paragraph_id, value in _as_string_key_dict(
            data.get("concept_ids_by_paragraph")
        ).items()
    }
    paragraph_ids_by_concept_data = _as_string_key_dict(
        data.get("paragraph_ids_by_concept", {})
    )
    concept_texts_by_id_data = _as_string_key_dict(data.get("concept_texts_by_id", {}))
    label_concept_ids_by_id_data = _as_string_key_dict(data.get("label_concept_ids_by_id", {}))

    paragraph_ids_by_concept = {
        concept_id: _as_string_list(value)
        for concept_id, value in paragraph_ids_by_concept_data.items()
    }
    if not paragraph_ids_by_concept:
        paragraph_ids_by_concept = _rebuild_paragraph_ids_by_concept(concept_ids_by_paragraph)

    concept_texts_by_id = {
        concept_id: _as_string(value)
        for concept_id, value in concept_texts_by_id_data.items()
    }
    if not concept_texts_by_id:
        concept_texts_by_id = _rebuild_concept_texts_by_id(paragraphs_by_id)

    label_concept_ids_by_id = {
        label_id: _as_string_list(value)
        for label_id, value in label_concept_ids_by_id_data.items()
    }
    paragraph_ids_by_label = {
        label_id: _as_string_list(value)
        for label_id, value in _as_string_key_dict(data.get("paragraph_ids_by_label")).items()
    }
    if not label_concept_ids_by_id:
        label_concept_ids_by_id = _rebuild_label_concept_ids_by_id(
            paragraph_ids_by_label,
            concept_ids_by_paragraph,
            fit_result,
        )

    return CorpusIndex(
        paragraphs_by_id=paragraphs_by_id,
        paragraph_ids_by_label=paragraph_ids_by_label,
        label_ids_by_paragraph={
            paragraph_id: _as_string_list(value)
            for paragraph_id, value in _as_string_key_dict(
                data.get("label_ids_by_paragraph")
            ).items()
        },
        concept_ids_by_paragraph=concept_ids_by_paragraph,
        paragraph_ids_by_concept=paragraph_ids_by_concept,
        label_display_names_by_id={
            label_id: _as_string(value)
            for label_id, value in _as_string_key_dict(
                data.get("label_display_names_by_id")
            ).items()
        },
        label_concept_ids_by_id=label_concept_ids_by_id,
        concept_texts_by_id=concept_texts_by_id,
    )


def _rebuild_paragraph_ids_by_concept(
    concept_ids_by_paragraph: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Rebuild concept reverse lookups from paragraph-side concept assignments."""

    paragraph_ids_by_concept: dict[str, list[str]] = {}
    for paragraph_id, concept_ids in concept_ids_by_paragraph.items():
        for concept_id in concept_ids:
            paragraph_ids_by_concept.setdefault(concept_id, []).append(paragraph_id)

    for concept_id in paragraph_ids_by_concept:
        paragraph_ids_by_concept[concept_id].sort()
    return paragraph_ids_by_concept


def _rebuild_concept_texts_by_id(
    paragraphs_by_id: dict[str, IndexedParagraph],
) -> dict[str, str]:
    """Rebuild concept text mappings from indexed paragraph records."""

    concept_texts_by_id: dict[str, str] = {}
    for paragraph in paragraphs_by_id.values():
        for concept_id, concept_text in zip(
            paragraph.concept_ids,
            paragraph.concept_texts,
            strict=False,
        ):
            concept_texts_by_id.setdefault(concept_id, concept_text)
    return concept_texts_by_id


def _rebuild_label_concept_ids_by_id(
    paragraph_ids_by_label: dict[str, list[str]],
    concept_ids_by_paragraph: dict[str, list[str]],
    fit_result: LabelGenerationResult | None = None,
) -> dict[str, list[str]]:
    """Rebuild label-to-concept mappings from paragraph-side assignments."""

    if fit_result is not None:
        return {
            community.id: sorted(community.concept_ids)
            for community in fit_result.communities
        }

    label_concept_ids_by_id: dict[str, list[str]] = {}
    for label_id, paragraph_ids in paragraph_ids_by_label.items():
        concept_ids = {
            concept_id
            for paragraph_id in paragraph_ids
            for concept_id in concept_ids_by_paragraph.get(paragraph_id, [])
        }
        label_concept_ids_by_id[label_id] = sorted(concept_ids)
    return label_concept_ids_by_id


def _as_string_list(value: object) -> list[str]:
    """Normalize a list of strings from serialized data."""

    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")

    normalized: list[str] = []
    for item in cast(list[object], value):
        normalized.append(_as_string(item))
    return normalized


def _as_string(value: object) -> str:
    """Normalize a string value from serialized data."""

    if not isinstance(value, str):
        raise TypeError("Expected a string value.")
    return value
