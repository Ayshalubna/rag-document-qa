import pytest
from langchain_core.documents import Document

from rag_qa.ingestion.chunking import chunk_documents


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "t.md"})


def test_chunks_respect_size_limit() -> None:
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_documents([_doc(text)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunks_carry_metadata_and_ids() -> None:
    chunks = chunk_documents([_doc("hello world. " * 100)], chunk_size=100, chunk_overlap=10)
    assert all(c.metadata["source"] == "t.md" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_overlap_preserves_continuity() -> None:
    text = "\n\n".join(f"Paragraph {i} content here." for i in range(50))
    chunks = chunk_documents([_doc(text)], chunk_size=120, chunk_overlap=40)
    assert len(chunks) >= 2


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_documents([_doc("x")], chunk_size=100, chunk_overlap=100)


def test_short_document_single_chunk() -> None:
    chunks = chunk_documents([_doc("short text")], chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"
