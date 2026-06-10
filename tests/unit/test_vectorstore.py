from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_qa.embeddings.fake import DeterministicFakeEmbeddings
from rag_qa.exceptions import IndexNotReadyError
from rag_qa.vectorstore.faiss_store import FaissVectorStore


def _store_with_docs() -> FaissVectorStore:
    store = FaissVectorStore(DeterministicFakeEmbeddings())
    store.add_documents(
        [
            Document(
                page_content="FAISS is a library for vector similarity search",
                metadata={"source": "faiss.md", "chunk_id": 0},
            ),
            Document(
                page_content="The chef baked a chocolate cake in the oven",
                metadata={"source": "cake.md", "chunk_id": 1},
            ),
        ]
    )
    return store


def test_search_before_ingest_raises() -> None:
    store = FaissVectorStore(DeterministicFakeEmbeddings())
    with pytest.raises(IndexNotReadyError):
        store.search("anything")


def test_search_ranks_relevant_chunk_first() -> None:
    store = _store_with_docs()
    results = store.search("vector similarity search with faiss", top_k=2)
    assert results[0].source == "faiss.md"
    assert results[0].score >= results[-1].score


def test_score_threshold_filters() -> None:
    store = _store_with_docs()
    all_results = store.search("faiss vector search", top_k=2, score_threshold=0.0)
    strict = store.search("faiss vector search", top_k=2, score_threshold=0.99)
    assert len(strict) <= len(all_results)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = _store_with_docs()
    index_dir = tmp_path / "idx"
    store.save(index_dir)

    fresh = FaissVectorStore(DeterministicFakeEmbeddings())
    assert fresh.load(index_dir) is True
    assert fresh.chunk_count == 2
    assert fresh.search("faiss", top_k=1)[0].source == "faiss.md"


def test_load_missing_index_returns_false(tmp_path: Path) -> None:
    store = FaissVectorStore(DeterministicFakeEmbeddings())
    assert store.load(tmp_path / "nope") is False
