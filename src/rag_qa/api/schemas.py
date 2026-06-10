"""API request/response models (the public contract, documented via OpenAPI)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What is FAISS?"])
    top_k: int | None = Field(None, gt=0, le=20, description="Override default top-k")
    score_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override default relevance threshold"
    )


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    latency_ms: float
    grounded: bool


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    index_ready: bool
    embedding_provider: str
    llm_model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    detail: str
