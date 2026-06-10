from __future__ import annotations

from pathlib import Path

import pytest

from rag_qa.config import Settings
from rag_qa.service import RAGService

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "env": "test",
        "embedding_provider": "fake",
        "llm_provider": "fake",
        "index_dir": tmp_path / "index",
        "upload_dir": tmp_path / "uploads",
        "score_threshold": 0.0,  # fake embeddings have a different score scale
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture()
def service(settings: Settings) -> RAGService:
    svc = RAGService(settings)
    svc.ingest(SAMPLE_DOCS, persist=False)
    return svc
