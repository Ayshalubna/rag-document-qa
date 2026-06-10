"""Application configuration.

All settings are overridable via environment variables with the ``RAG_`` prefix
(e.g. ``RAG_TOP_K=6``) or a local ``.env`` file. Defaults reflect the values
selected by the evaluation harness (see README, "Evaluation methodology").
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Runtime
    env: Literal["development", "production", "test"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Retrieval tuning
    chunk_size: int = Field(512, gt=0, description="Characters per chunk")
    chunk_overlap: int = Field(64, ge=0)
    top_k: int = Field(4, gt=0, le=50)
    score_threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Minimum relevance (1 - normalised L2 distance) for a chunk to be used.",
    )

    # Embeddings
    embedding_provider: Literal["huggingface", "fake"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_provider: Literal["ollama", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_timeout_s: int = Field(60, gt=0)

    # Paths / limits
    index_dir: Path = Path("data/index")
    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MiB

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info: object) -> int:
        # chunk_size validates before chunk_overlap (declaration order)
        data = getattr(info, "data", {})
        size = data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
