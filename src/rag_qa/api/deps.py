from __future__ import annotations

from fastapi import Request

from rag_qa.service import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service  # type: ignore[no-any-return]
