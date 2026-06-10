"""End-to-end API tests with fake embedding/LLM backends (no model downloads)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import SAMPLE_DOCS, make_settings

from rag_qa.api.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def _ingest_sample(client: TestClient) -> None:
    doc = SAMPLE_DOCS / "faiss_overview.md"
    with doc.open("rb") as fh:
        response = client.post("/api/v1/documents", files={"file": (doc.name, fh, "text/markdown")})
    assert response.status_code == 201, response.text


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_503_before_ingest(client: TestClient) -> None:
    assert client.get("/ready").status_code == 503


def test_query_409_before_ingest(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": "What is FAISS?"})
    assert response.status_code == 409


def test_upload_then_query_returns_cited_answer(client: TestClient) -> None:
    _ingest_sample(client)
    assert client.get("/ready").status_code == 200

    response = client.post(
        "/api/v1/query", json={"question": "What is FAISS used for in similarity search?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["source"] == "faiss_overview.md"
    assert body["latency_ms"] >= 0
    assert "x-request-id" in response.headers


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents", files={"file": ("evil.exe", b"MZ", "application/octet-stream")}
    )
    assert response.status_code == 400


def test_query_validation(client: TestClient) -> None:
    assert client.post("/api/v1/query", json={"question": "hi"}).status_code == 422
    assert client.post("/api/v1/query", json={}).status_code == 422


def test_stats(client: TestClient) -> None:
    _ingest_sample(client)
    body = client.get("/api/v1/stats").json()
    assert body["index_ready"] is True
    assert body["chunks"] > 0
