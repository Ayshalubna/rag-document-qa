"""CLI: run the eval harness.  Usage: rag-eval --dataset eval/datasets/sample_eval.jsonl"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_qa.config import get_settings
from rag_qa.eval.runner import run_eval
from rag_qa.logging_conf import configure_logging
from rag_qa.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL eval dataset")
    parser.add_argument("--report", type=Path, default=None, help="Write JSON report here")
    parser.add_argument(
        "--ingest", type=Path, default=None, help="Optionally (re)ingest this path first"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    service = RAGService.from_settings(settings)
    if args.ingest:
        service.ingest(args.ingest, persist=False)
    report = run_eval(service, args.dataset)
    print(report.to_json())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
