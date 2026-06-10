# ---- builder ----
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install ".[inference]"

# ---- runtime ----
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000
RUN useradd --create-home appuser
COPY --from=builder /install /usr/local
COPY data/sample_docs ./data/sample_docs
RUN mkdir -p data/index data/uploads && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"
CMD ["uvicorn", "rag_qa.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
