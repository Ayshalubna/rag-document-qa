"""RAG question-answering chain.

Pipeline: query -> vector retrieval (top-k, threshold) -> grounded prompt ->
local LLM -> answer + citations. The prompt constrains the model to the
retrieved context, the primary lever for hallucination control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rag_qa.llm.base import LLM
from rag_qa.vectorstore.faiss_store import FaissVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are a careful assistant answering questions strictly from the provided context.

Rules:
- Use ONLY the context below. Do not use outside knowledge.
- Cite supporting passages inline with their bracketed numbers, e.g. [1], [2].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

_NO_CONTEXT_ANSWER = "I don't know based on the provided documents."


@dataclass(frozen=True)
class QAResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    grounded: bool = True  # False when retrieval found nothing usable


class RagQAChain:
    def __init__(self, store: FaissVectorStore, llm: LLM) -> None:
        self._store = store
        self._llm = llm

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            origin = chunk.source + (f", p.{chunk.page}" if chunk.page else "")
            blocks.append(f"[{i}] ({origin})\n{chunk.text}")
        return "\n\n".join(blocks)

    def ask(self, question: str, top_k: int = 4, score_threshold: float = 0.35) -> QAResult:
        start = time.perf_counter()
        chunks = self._store.search(question, top_k=top_k, score_threshold=score_threshold)

        if not chunks:
            # Refuse rather than let the model free-associate: anti-hallucination guard.
            return QAResult(
                answer=_NO_CONTEXT_ANSWER,
                sources=[],
                model=self._llm.model_name,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
                grounded=False,
            )

        prompt = _PROMPT_TEMPLATE.format(
            context=self._format_context(chunks), question=question.strip()
        )
        answer = self._llm.generate(prompt).strip()
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info("Answered in %.1f ms using %d chunk(s)", latency_ms, len(chunks))
        return QAResult(
            answer=answer,
            sources=chunks,
            model=self._llm.model_name,
            latency_ms=latency_ms,
            grounded=True,
        )
