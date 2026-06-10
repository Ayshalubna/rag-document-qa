from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from rag_qa import __version__
from rag_qa.api.deps import get_rag_service
from rag_qa.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StatsResponse,
)
from rag_qa.exceptions import (
    EmptyDocumentError,
    IndexNotReadyError,
    LLMUnavailableError,
    UnsupportedFileTypeError,
)
from rag_qa.ingestion import SUPPORTED_EXTENSIONS
from rag_qa.service import RAGService

router = APIRouter()
Service = Annotated[RAGService, Depends(get_rag_service)]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(version=__version__)


@router.get("/ready", tags=["ops"], responses={503: {"model": ErrorResponse}})
def ready(service: Service) -> dict[str, bool]:
    """Readiness probe: index must contain at least one document."""
    if not service.store.is_ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="Index empty; ingest documents first."
        )
    return {"ready": True}


@router.get("/api/v1/stats", response_model=StatsResponse, tags=["ops"])
def stats(service: Service) -> StatsResponse:
    return StatsResponse(
        documents=service.store.doc_count,
        chunks=service.store.chunk_count,
        index_ready=service.store.is_ready,
        embedding_provider=service.settings.embedding_provider,
        llm_model=service.settings.ollama_model
        if service.settings.llm_provider == "ollama"
        else service.settings.llm_provider,
    )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["qa"],
    responses={409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def query(body: QueryRequest, service: Service) -> QueryResponse:
    """Answer a question grounded in the indexed corpus, with citations."""
    try:
        result = service.ask(body.question, body.top_k, body.score_threshold)
    except IndexNotReadyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return QueryResponse(
        answer=result.answer,
        model=result.model,
        latency_ms=result.latency_ms,
        grounded=result.grounded,
        sources=[
            SourceChunk(
                source=s.source,
                page=s.page,
                score=s.score,
                snippet=s.text[:300],
            )
            for s in result.sources
        ],
    )


@router.post(
    "/api/v1/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["qa"],
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_document(file: UploadFile, service: Service) -> IngestResponse:
    """Upload and index a document (txt, md, rst, pdf)."""
    filename = Path(file.filename or "upload").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type; supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    content = await file.read()
    if len(content) > service.settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {service.settings.max_upload_bytes} bytes.",
        )
    upload_dir = service.settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    dest.write_bytes(content)
    try:
        chunks = service.ingest(dest)
    except (UnsupportedFileTypeError, EmptyDocumentError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return IngestResponse(filename=filename, chunks_indexed=chunks)
