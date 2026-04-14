"""Tests for embedding provider runtime behavior."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from labelrag import EmbeddingConfig, SentenceTransformerEmbeddingProvider


def test_sentence_transformer_provider_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sentence-transformers should raise a useful runtime error."""

    real_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "sentence_transformers":
            raise ImportError("missing dependency")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    provider = SentenceTransformerEmbeddingProvider(EmbeddingConfig())

    with pytest.raises(RuntimeError, match="Install labelrag"):
        provider.embed_query("language models")


def test_sentence_transformer_provider_wraps_model_load_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model loading failures should mention the configured model name."""

    class FailingSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            raise OSError(f"cannot load {model_name}")

    fake_module = SimpleNamespace(SentenceTransformer=FailingSentenceTransformer)
    real_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "sentence_transformers":
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    provider = SentenceTransformerEmbeddingProvider(
        EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    )

    with pytest.raises(RuntimeError, match="all-MiniLM-L6-v2"):
        provider.embed_documents(["language models for developers"])
