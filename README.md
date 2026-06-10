# RAG Document Q&A

Production-ready **Retrieval-Augmented Generation** system for question answering over your own documents — fully local, zero external API dependency.

**Stack:** Python · LangChain · FAISS · Hugging Face sentence-transformers · Ollama · FastAPI · Docker

[![CI](https://github.com/Ayshalubna/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Ayshalubna/rag-document-qa/actions)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Why this exists

LLMs hallucinate and know nothing about your private documents. This service grounds every answer in retrieved passages from your own corpus, cites its sources, and **refuses to answer** when retrieval comes back empty — and because embeddings and generation both run locally (sentence-transformers + Ollama), no text ever leaves your machine.

## Architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │                   FastAPI                        │
                 │  POST /api/v1/documents      POST /api/v1/query  │
                 └──────────┬───────────────────────────┬───────────┘
                            │                           │
                   ┌────────▼────────┐         ┌────────▼────────┐
   txt/md/pdf ───► │   Ingestion     │         │   RAG QA chain  │
                   │ load → chunk    │         │ retrieve → prompt│
                   └────────┬────────┘         │ → generate+cite │
                            │                  └───┬─────────┬───┘
                   ┌────────▼────────┐             │         │
                   │  HF embeddings  │◄────────────┘   ┌─────▼─────┐
                   │ (all-MiniLM-L6) │                 │  Ollama   │
                   └────────┬────────┘                 │ llama3.1  │
                            │                          └───────────┘
                   ┌────────▼────────┐
                   │   FAISS index   │  (persisted to disk, atomic writes)
                   └─────────────────┘
```

Design notes:

- **Hexagonal-ish layering.** `ingestion`, `embeddings`, `vectorstore`, `llm`, and `chain` are independent modules wired together in `service.py`; the API layer only talks to `RAGService`. Embedding and LLM backends are selected by factory and satisfy small interfaces (`Embeddings`, an `LLM` protocol), so tests inject deterministic fakes — **no model downloads in CI**, full pipeline still exercised.
- **Hallucination control.** Three layers: (1) a strict grounding prompt that forbids outside knowledge and demands inline citations, (2) a relevance `score_threshold` that discards weak retrievals, (3) a hard refusal path — if nothing passes the threshold the chain returns "I don't know" without ever calling the LLM.
- **Operational hygiene.** Structured JSON logs with request IDs and latency, liveness (`/health`) and readiness (`/ready`) probes, typed domain exceptions mapped to proper HTTP codes, atomic index persistence, non-root Docker runtime, multi-stage image.

## Quick start

### Docker (recommended)

```bash
docker compose up --build          # starts Ollama + the API
docker compose exec ollama ollama pull llama3.1:8b
curl -X POST localhost:8000/api/v1/documents -F "file=@data/sample_docs/faiss_overview.md"
curl -X POST localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is FAISS and who developed it?"}'
```

### Local

```bash
# 1. Install (inference extra pulls in sentence-transformers + langchain-ollama)
pip install -e ".[dev,inference]"

# 2. Run a local LLM
ollama pull llama3.1:8b && ollama serve

# 3. Index documents and serve
rag-ingest --source data/sample_docs
make run        # uvicorn rag_qa.api.main:create_app --factory
```

Interactive API docs: <http://localhost:8000/docs>

## API

| Method | Path                | Description                                  |
| ------ | ------------------- | -------------------------------------------- |
| POST   | `/api/v1/query`     | Ask a question; returns answer + citations   |
| POST   | `/api/v1/documents` | Upload & index a document (txt, md, rst, pdf)|
| GET    | `/api/v1/stats`     | Corpus / index statistics                    |
| GET    | `/health`           | Liveness probe                               |
| GET    | `/ready`            | Readiness probe (503 until index is non-empty)|

Example response:

```json
{
  "answer": "FAISS is a library for efficient similarity search of dense vectors, developed by Meta AI Research [1].",
  "sources": [
    {"source": "faiss_overview.md", "page": null, "score": 0.78, "snippet": "FAISS (Facebook AI Similarity Search) is a library..."}
  ],
  "model": "llama3.1:8b",
  "latency_ms": 1240.5,
  "grounded": true
}
```

`question` accepts optional `top_k` and `score_threshold` overrides per request.

## Configuration

Everything is a `RAG_`-prefixed environment variable (or `.env`); see [`.env.example`](.env.example). Key knobs:

| Variable              | Default                  | Notes                                        |
| --------------------- | ------------------------ | -------------------------------------------- |
| `RAG_CHUNK_SIZE`      | `512`                    | chars per chunk                              |
| `RAG_CHUNK_OVERLAP`   | `64`                     | must be < chunk size                         |
| `RAG_TOP_K`           | `4`                      | retrieved chunks per query                   |
| `RAG_SCORE_THRESHOLD` | `0.35`                   | min relevance (1 − normalised L2 distance)   |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2`       | any sentence-transformers model              |
| `RAG_OLLAMA_MODEL`    | `llama3.1:8b`            | any model served by Ollama                   |

## Evaluation methodology

The eval harness (`rag-eval`) replays a labelled JSONL dataset (`{question, expected_source, answer_keywords}`) through the full pipeline and reports:

- **Retrieval:** `hit_rate@k` (expected source retrieved) and `MRR` (rank of first relevant chunk).
- **Generation:** `faithfulness_proxy` — fraction of answer tokens supported by the retrieved context (low = likely hallucination) — and `keyword_recall` against gold keywords.
- **Latency:** p50 / p95 end-to-end, asserted against the **sub-2s SLO** in CI.

```bash
rag-eval --dataset eval/datasets/sample_eval.jsonl --report eval/reports/latest.json
```

Defaults (`chunk_size=512`, `overlap=64`, `top_k=4`, `threshold=0.35`) were chosen by sweeping these parameters with the harness: smaller chunks improved precision but fragmented context (faithfulness dropped); larger `top_k` raised hit rate marginally while diluting the prompt and increasing latency and hallucination rate. The harness also runs in CI with deterministic fake backends as a quality regression gate (`tests/integration/test_eval_harness.py`).

## Development

```bash
make install     # editable install + dev tools
make lint        # ruff check + format check
make type        # mypy (strict-ish)
make test        # pytest: 38 unit + integration tests
make cov         # with coverage
pre-commit install
```

CI (GitHub Actions) runs lint, type check, tests on Python 3.10–3.12, and a Docker build on every push/PR.

## Project layout

```
src/rag_qa/
├── config.py          # pydantic-settings, RAG_* env vars
├── service.py         # composition root (ingest + ask)
├── ingestion/         # loaders (txt/md/rst/pdf) + recursive chunking
├── embeddings/        # HF sentence-transformers | deterministic fake
├── vectorstore/       # FAISS wrapper: thread-safe, atomic persistence
├── llm/               # Ollama client (retry/backoff) | fake, behind a Protocol
├── chain/             # grounded prompt, citations, refusal path
├── api/               # FastAPI app factory, routes, schemas, middleware
├── eval/              # metrics + harness
└── cli/               # rag-ingest, rag-eval
tests/                 # unit + integration (fake backends, no downloads)
```

## License

MIT
