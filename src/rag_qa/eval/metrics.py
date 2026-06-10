"""Evaluation metrics.

Retrieval quality:
- hit_rate@k: fraction of questions whose expected source appears in retrieval.
- MRR: mean reciprocal rank of the first relevant retrieved chunk.

Generation quality (reference-free proxies, no judge model needed in CI):
- faithfulness_proxy: fraction of answer tokens supported by retrieved context;
  low values flag likely hallucination.
- keyword_recall: fraction of gold answer keywords present in the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "not",
        "don",
        "know",
        "based",
        "provided",
        "documents",
    ]
)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(frozen=True)
class EvalSample:
    question: str
    expected_source: str
    answer_keywords: list[str] = field(default_factory=list)


def hit_rate(retrieved_sources: list[list[str]], expected: list[str]) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for sources, exp in zip(retrieved_sources, expected, strict=True) if exp in sources)
    return hits / len(expected)


def mrr(retrieved_sources: list[list[str]], expected: list[str]) -> float:
    if not expected:
        return 0.0
    total = 0.0
    for sources, exp in zip(retrieved_sources, expected, strict=True):
        for rank, source in enumerate(sources, start=1):
            if source == exp:
                total += 1.0 / rank
                break
    return total / len(expected)


def faithfulness_proxy(answer: str, context: str) -> float:
    """Share of (non-stopword) answer tokens that appear in the retrieved context."""
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = _tokens(context)
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in answer_lower) / len(keywords)
