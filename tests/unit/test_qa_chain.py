from langchain_core.documents import Document

from rag_qa.chain.qa_chain import RagQAChain
from rag_qa.embeddings.fake import DeterministicFakeEmbeddings
from rag_qa.llm.fake import EchoLLM
from rag_qa.vectorstore.faiss_store import FaissVectorStore


def _chain() -> RagQAChain:
    store = FaissVectorStore(DeterministicFakeEmbeddings())
    store.add_documents(
        [
            Document(
                page_content="FAISS performs efficient vector similarity search.",
                metadata={"source": "faiss.md", "chunk_id": 0},
            )
        ]
    )
    return RagQAChain(store, EchoLLM())


def test_grounded_answer_includes_sources_and_latency() -> None:
    result = _chain().ask("what does faiss do for similarity search", score_threshold=0.0)
    assert result.grounded is True
    assert result.sources and result.sources[0].source == "faiss.md"
    assert result.latency_ms >= 0
    assert result.model == "fake-echo"


def test_refuses_when_nothing_retrieved() -> None:
    # threshold of 1.0 filters out everything -> chain must refuse, not hallucinate
    result = _chain().ask("unrelated question", score_threshold=1.0)
    assert result.grounded is False
    assert result.sources == []
    assert "don't know" in result.answer
