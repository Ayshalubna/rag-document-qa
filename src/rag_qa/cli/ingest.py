"""CLI: index a file or directory.  Usage: rag-ingest --source data/sample_docs"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_qa.config import get_settings
from rag_qa.logging_conf import configure_logging
from rag_qa.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the FAISS index.")
    parser.add_argument("--source", type=Path, required=True, help="File or directory to index")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    service = RAGService.from_settings(settings)
    added = service.ingest(args.source)
    print(f"Indexed {added} chunk(s) from {args.source}; index saved to {settings.index_dir}")


if __name__ == "__main__":
    main()
