"""Eval harness: replay a labelled dataset through the full RAG pipeline.

Used to (a) regression-test quality in CI with fake backends, and (b) tune
chunk_size / top_k / score_threshold against real backends locally.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_qa.eval.metrics import EvalSample, faithfulness_proxy, hit_rate, keyword_recall, mrr
from rag_qa.service import RAGService


@dataclass(frozen=True)
class EvalReport:
    samples: int
    hit_rate_at_k: float
    mrr: float
    mean_faithfulness: float
    mean_keyword_recall: float
    p50_latency_ms: float
    p95_latency_ms: float
    config: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_dataset(path: Path) -> list[EvalSample]:
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                EvalSample(
                    question=row["question"],
                    expected_source=row["expected_source"],
                    answer_keywords=row.get("answer_keywords", []),
                )
            )
    return samples


def run_eval(service: RAGService, dataset: Path) -> EvalReport:
    samples = load_dataset(dataset)
    retrieved: list[list[str]] = []
    expected: list[str] = []
    faiths: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []

    for sample in samples:
        result = service.ask(sample.question)
        sources = [s.source for s in result.sources]
        retrieved.append(sources)
        expected.append(sample.expected_source)
        context = "\n".join(s.text for s in result.sources)
        faiths.append(faithfulness_proxy(result.answer, context))
        recalls.append(keyword_recall(result.answer, sample.answer_keywords))
        latencies.append(result.latency_ms)

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, round(p * (len(ordered) - 1)))
        return ordered[int(idx)]

    return EvalReport(
        samples=len(samples),
        hit_rate_at_k=round(hit_rate(retrieved, expected), 4),
        mrr=round(mrr(retrieved, expected), 4),
        mean_faithfulness=round(statistics.fmean(faiths), 4) if faiths else 0.0,
        mean_keyword_recall=round(statistics.fmean(recalls), 4) if recalls else 0.0,
        p50_latency_ms=pct(latencies, 0.5),
        p95_latency_ms=pct(latencies, 0.95),
        config={
            "chunk_size": service.settings.chunk_size,
            "chunk_overlap": service.settings.chunk_overlap,
            "top_k": service.settings.top_k,
            "score_threshold": service.settings.score_threshold,
            "embedding_provider": service.settings.embedding_provider,
            "llm_provider": service.settings.llm_provider,
        },
    )
