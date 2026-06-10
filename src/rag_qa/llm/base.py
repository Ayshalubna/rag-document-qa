"""LLM interface (structural typing keeps backends swappable and testable)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    model_name: str

    def generate(self, prompt: str) -> str:
        """Generate a completion for the prompt. Raises LLMUnavailableError on failure."""
        ...
