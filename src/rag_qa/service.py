"""RAGService: composition root wiring ingestion, store, and chain together."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_qa.chain.qa_chain import QAResult, RagQAChain
from rag_qa.config import Settings
from rag_qa.embeddings import build_embeddings
from rag_qa.ingestion import chunk_documents, load_path
from rag_qa.llm import build_llm
from rag_qa.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = FaissVectorStore(build_embeddings(settings))
        self.chain = RagQAChain(self.store, build_llm(settings))

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGService:
        service = cls(settings)
        if service.store.load(settings.index_dir):
            logger.info("Restored persisted index")
        return service

    def ingest(self, path: Path, persist: bool = True) -> int:
        """Load, chunk, embed and index a file or directory. Returns chunks added."""
        documents = load_path(path)
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.store.add_documents(chunks)
        if persist and added:
            self.store.save(self.settings.index_dir)
        return added

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> QAResult:
        return self.chain.ask(
            question,
            top_k=top_k or self.settings.top_k,
            score_threshold=(
                score_threshold if score_threshold is not None else self.settings.score_threshold
            ),
        )
