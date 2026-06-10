"""Ollama-backed LLM (fully local inference — no external API, full data privacy)."""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_qa.exceptions import LLMUnavailableError

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        timeout_s: int = 60,
    ) -> None:
        from langchain_ollama import ChatOllama  # lazy: optional 'inference' extra

        self.model_name = model
        self._client = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout_s,
        )

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return str(response.content)

    def generate(self, prompt: str) -> str:
        try:
            return self._invoke(prompt)
        except Exception as exc:
            logger.exception("Ollama generation failed")
            raise LLMUnavailableError(f"LLM backend error: {exc}") from exc
