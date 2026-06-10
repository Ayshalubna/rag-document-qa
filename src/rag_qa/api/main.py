"""FastAPI application factory.

Run with: uvicorn rag_qa.api.main:create_app --factory
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_qa import __version__
from rag_qa.api.middleware import request_context_middleware
from rag_qa.api.routes import router
from rag_qa.config import Settings, get_settings
from rag_qa.logging_conf import configure_logging
from rag_qa.service import RAGService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag_service = RAGService.from_settings(settings)
        logger.info(
            "Service started (env=%s, embeddings=%s, llm=%s)",
            settings.env,
            settings.embedding_provider,
            settings.llm_provider,
        )
        yield

    app = FastAPI(
        title="RAG Document Q&A",
        version=__version__,
        description="Retrieval-Augmented Generation over your documents. "
        "Fully local inference: FAISS + sentence-transformers + Ollama.",
        lifespan=lifespan,
    )
    app.middleware("http")(request_context_middleware)
    app.include_router(router)
    return app
