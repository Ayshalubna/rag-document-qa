import pytest
from pydantic import ValidationError

from rag_qa.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.chunk_size == 512
    assert s.top_k == 4
    assert s.embedding_provider == "huggingface"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "fake")
    s = Settings(_env_file=None)
    assert s.top_k == 7
    assert s.llm_provider == "fake"


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_provider="openai")  # type: ignore[arg-type]
