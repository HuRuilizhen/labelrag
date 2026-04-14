"""Embedding persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ParagraphEmbeddingStore:
    """Persisted paragraph embedding matrix and metadata."""

    paragraph_ids: list[str]
    matrix: np.ndarray
    provider_name: str
    model_name: str
    normalized: bool

    def save(self, path: str | Path) -> None:
        """Persist the embedding store to a `.npz` artifact."""

        destination = Path(path)
        np.savez(
            destination,
            paragraph_ids=np.asarray(self.paragraph_ids, dtype=str),
            matrix=np.asarray(self.matrix, dtype=np.float32),
            provider_name=np.asarray(self.provider_name),
            model_name=np.asarray(self.model_name),
            normalized=np.asarray(self.normalized),
        )


def load_paragraph_embedding_store(path: str | Path) -> ParagraphEmbeddingStore:
    """Load a persisted paragraph embedding store from disk."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        paragraph_ids = [str(value) for value in data["paragraph_ids"].tolist()]
        matrix = np.asarray(data["matrix"], dtype=np.float32)
        provider_name = str(data["provider_name"].tolist())
        model_name = str(data["model_name"].tolist())
        normalized = bool(data["normalized"].tolist())
    return ParagraphEmbeddingStore(
        paragraph_ids=paragraph_ids,
        matrix=matrix,
        provider_name=provider_name,
        model_name=model_name,
        normalized=normalized,
    )
