"""Document chunking.

Uses a recursive character splitter that prefers paragraph, then sentence,
then word boundaries — keeping semantically coherent units together. Chunk
size / overlap defaults were selected with the eval harness (see README).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into overlapping chunks, tagging each with a chunk id."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
