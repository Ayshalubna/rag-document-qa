"""FAISS vector store wrapper.

Thin, typed facade over LangChain's FAISS integration adding:
- thread-safe add/search (FAISS writes are not thread-safe),
- atomic persistence to disk (write to temp dir, then rename),
- relevance scores normalised to [0, 1] with threshold filtering.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_qa.exceptions import IndexNotReadyError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float  # relevance in [0, 1]; higher is better
    page: int | None = None
    chunk_id: int | None = None


class FaissVectorStore:
    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._store: FAISS | None = None
        self._lock = threading.Lock()
        self._doc_count = 0

    # -- properties -------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._store is not None

    @property
    def chunk_count(self) -> int:
        return self._store.index.ntotal if self._store is not None else 0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    # -- mutation ---------------------------------------------------------
    def add_documents(self, chunks: list[Document]) -> int:
        """Embed and index chunks. Returns number of chunks added."""
        if not chunks:
            return 0
        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(chunks, self._embeddings)
            else:
                self._store.add_documents(chunks)
            self._doc_count += len({c.metadata.get("source") for c in chunks})
        logger.info("Indexed %d chunk(s); index size now %d", len(chunks), self.chunk_count)
        return len(chunks)

    # -- search -----------------------------------------------------------
    def search(
        self, query: str, top_k: int = 4, score_threshold: float = 0.0
    ) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks at or above score_threshold."""
        if self._store is None:
            raise IndexNotReadyError("No documents indexed yet; ingest documents first.")
        with self._lock:
            results = self._store.similarity_search_with_score(query, k=top_k)
        retrieved = []
        for doc, distance in results:
            # FAISS returns L2 distance on normalised vectors: d^2 in [0, 4].
            # Map to a monotonic relevance score in [0, 1].
            relevance = max(0.0, 1.0 - float(distance) / 2.0)
            if relevance < score_threshold:
                continue
            retrieved.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    page=doc.metadata.get("page"),
                    chunk_id=doc.metadata.get("chunk_id"),
                    score=round(relevance, 4),
                )
            )
        return retrieved

    # -- persistence ------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        if self._store is None:
            raise IndexNotReadyError("Nothing to save: index is empty.")
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tmp = Path(tempfile.mkdtemp(prefix="faiss-", dir=str(index_dir.parent)))
            try:
                self._store.save_local(str(tmp))
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                tmp.rename(index_dir)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
        logger.info("Index persisted to %s", index_dir)

    def load(self, index_dir: Path) -> bool:
        """Load a persisted index. Returns False if none exists."""
        if not (index_dir / "index.faiss").exists():
            return False
        with self._lock:
            self._store = FAISS.load_local(
                str(index_dir),
                self._embeddings,
                # Safe here: we only ever load indexes this service wrote.
                allow_dangerous_deserialization=True,
            )
        logger.info("Loaded index from %s (%d chunks)", index_dir, self.chunk_count)
        return True
