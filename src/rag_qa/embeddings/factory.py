"""Embedding backend factory."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag_qa.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """Return the configured embedding backend.

    ``huggingface`` runs sentence-transformers fully locally (no external API).
    ``fake`` is a deterministic, dependency-free backend for tests and CI.
    """
    if settings.embedding_provider == "fake":
        from rag_qa.embeddings.fake import DeterministicFakeEmbeddings

        return DeterministicFakeEmbeddings()

    from langchain_huggingface import HuggingFaceEmbeddings  # heavy import, keep lazy

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
