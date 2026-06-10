"""File loaders.

Each loader returns LangChain ``Document`` objects with ``source`` (and where
available ``page``) metadata, which the QA chain later surfaces as citations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from langchain_core.documents import Document

from rag_qa.exceptions import EmptyDocumentError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader  # lazy: keeps base import light

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": i + 1}))
    return docs


_LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".txt": _load_text,
    ".md": _load_text,
    ".rst": _load_text,
    ".pdf": _load_pdf,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load_file(path: Path) -> list[Document]:
    """Load a single file into Documents.

    Raises:
        UnsupportedFileTypeError: extension has no registered loader.
        EmptyDocumentError: file produced no extractable text.
    """
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {path.suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    docs = [d for d in loader(path) if d.page_content.strip()]
    if not docs:
        raise EmptyDocumentError(f"No extractable text in {path.name}")
    logger.info("Loaded %s (%d document section(s))", path.name, len(docs))
    return docs


def load_path(path: Path) -> list[Document]:
    """Load a file, or recursively load every supported file in a directory."""
    if path.is_file():
        return load_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    docs: list[Document] = []
    skipped: list[str] = []
    files: Iterable[Path] = sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(load_file(file))
            except EmptyDocumentError:
                skipped.append(file.name)
        else:
            skipped.append(file.name)
    if skipped:
        logger.info("Skipped %d unsupported/empty file(s): %s", len(skipped), skipped[:10])
    return docs
