"""Quality regression gate: the eval harness itself runs in CI with fakes."""

from pathlib import Path

import pytest

from rag_qa.eval.runner import run_eval
from rag_qa.service import RAGService

pytestmark = pytest.mark.integration

DATASET = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "sample_eval.jsonl"


def test_eval_report_quality_floor(service: RAGService) -> None:
    report = run_eval(service, DATASET)
    assert report.samples == 5
    # Even with fake bag-of-words embeddings, lexical retrieval should mostly hit.
    assert report.hit_rate_at_k >= 0.6
    assert report.mrr >= 0.5
    assert 0.0 <= report.mean_faithfulness <= 1.0
    assert report.p95_latency_ms < 2000  # latency SLO from the README
    json_text = report.to_json()
    assert "hit_rate_at_k" in json_text
