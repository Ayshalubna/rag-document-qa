"""Fake LLM for tests/CI: extractive answer over the supplied context."""

from __future__ import annotations


class EchoLLM:
    model_name = "fake-echo"

    def generate(self, prompt: str) -> str:
        # Extract context lines injected by the QA prompt and parrot the first
        # sentence — deterministic, grounded-by-construction output for tests.
        marker = "Context:"
        if marker in prompt:
            context = prompt.split(marker, 1)[1].split("Question:", 1)[0]
            for line in context.splitlines():
                line = line.strip()
                if line and not line.startswith("["):
                    return line.split(". ")[0] + " [1]"
        return "I don't know based on the provided documents."
