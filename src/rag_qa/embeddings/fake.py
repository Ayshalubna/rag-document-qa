"""Deterministic fake embeddings for tests/CI.

Token-hash bag-of-words vectors: texts sharing vocabulary get higher cosine
similarity, so retrieval order is meaningful in tests without downloading a
model. Never use in production.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicFakeEmbeddings(Embeddings):
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
