from __future__ import annotations

from rag_qa.config import Settings
from rag_qa.llm.base import LLM


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        from rag_qa.llm.fake import EchoLLM

        return EchoLLM()

    from rag_qa.llm.ollama_client import OllamaLLM

    return OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
    )
