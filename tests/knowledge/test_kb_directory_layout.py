from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from deeptutor.knowledge.add_documents import DocumentAdder
import deeptutor.knowledge.initializer as initializer_module
from deeptutor.knowledge.initializer import KnowledgeBaseInitializer


def test_initializer_creates_raw_only_source_layout(tmp_path: Path) -> None:
    initializer = KnowledgeBaseInitializer(kb_name="demo", base_dir=str(tmp_path))
    initializer.create_directory_structure()

    kb_dir = tmp_path / "demo"
    assert (kb_dir / "raw").exists()
    assert not (kb_dir / "llamaindex_storage").exists()
    assert not (kb_dir / "index_versions").exists()
    assert not (kb_dir / "images").exists()
    assert not (kb_dir / "content_list").exists()
    assert not (kb_dir / "rag_storage").exists()


def test_document_adder_does_not_create_compatibility_dirs(tmp_path: Path) -> None:
    kb_dir = tmp_path / "demo"
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)
    (kb_dir / "version-1").mkdir(parents=True, exist_ok=True)
    (kb_dir / "version-1" / "docstore.json").write_text("{}", encoding="utf-8")
    (kb_dir / "version-1" / "index_store.json").write_text("{}", encoding="utf-8")
    (kb_dir / "version-1" / "meta.json").write_text(
        '{"signature": "sig", "version": "version-1"}',
        encoding="utf-8",
    )

    DocumentAdder(kb_name="demo", base_dir=str(tmp_path))

    assert (kb_dir / "raw").exists()
    assert (kb_dir / "version-1").exists()
    assert not (kb_dir / "llamaindex_storage").exists()
    assert not (kb_dir / "index_versions").exists()
    assert not (kb_dir / "images").exists()
    assert not (kb_dir / "content_list").exists()


def test_initializer_records_initial_document_hashes_after_index_success(
    monkeypatch, tmp_path: Path
) -> None:
    captured_paths: list[str] = []

    class _RagService:
        def __init__(self, **_kwargs) -> None:
            pass

        async def initialize(self, *, file_paths, **_kwargs) -> bool:
            captured_paths.extend(file_paths)
            return True

    monkeypatch.setattr(initializer_module, "RAGService", _RagService)
    initializer = KnowledgeBaseInitializer(
        kb_name="demo",
        base_dir=str(tmp_path),
        rag_provider="lightrag",
    )
    initializer.create_directory_structure()
    nested = initializer.raw_dir / "part-1"
    nested.mkdir()
    first = initializer.raw_dir / "chapter-1.pdf"
    second = nested / "chapter-2.pdf"
    first.write_bytes(b"first document")
    second.write_bytes(b"second document")

    assert asyncio.run(initializer.process_documents()) is True

    metadata = json.loads((initializer.kb_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["file_hashes"] == {
        "chapter-1.pdf": hashlib.sha256(first.read_bytes()).hexdigest(),
        "part-1/chapter-2.pdf": hashlib.sha256(second.read_bytes()).hexdigest(),
    }
    assert set(captured_paths) == {str(first), str(second)}


def test_initializer_does_not_record_hashes_when_indexing_fails(
    monkeypatch, tmp_path: Path
) -> None:
    class _RagService:
        def __init__(self, **_kwargs) -> None:
            pass

        async def initialize(self, **_kwargs) -> bool:
            return False

    monkeypatch.setattr(initializer_module, "RAGService", _RagService)
    initializer = KnowledgeBaseInitializer(kb_name="demo", base_dir=str(tmp_path))
    initializer.create_directory_structure()
    (initializer.raw_dir / "chapter.pdf").write_bytes(b"document")

    with pytest.raises(RuntimeError, match="RAG pipeline returned failure"):
        asyncio.run(initializer.process_documents())

    metadata = json.loads((initializer.kb_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "file_hashes" not in metadata
