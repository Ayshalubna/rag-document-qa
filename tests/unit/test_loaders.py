from pathlib import Path

import pytest

from rag_qa.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from rag_qa.ingestion.loaders import load_file, load_path


def test_load_text_file(tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")
    docs = load_file(f)
    assert docs[0].page_content == "hello world"
    assert docs[0].metadata["source"] == "note.txt"


def test_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileTypeError):
        load_file(f)


def test_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_file(f)


def test_load_directory_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    docs = load_path(tmp_path)
    assert len(docs) == 1


def test_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "ghost")
