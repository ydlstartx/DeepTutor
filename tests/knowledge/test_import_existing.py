from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptutor.knowledge.import_existing import (
    import_existing_knowledge_base,
    list_upload_folders,
    probe_import_folder,
    resolve_upload_folder,
)
from deeptutor.knowledge.manager import KnowledgeBaseManager


def _write_ready_upload(data_root: Path, name: str = "高中课程") -> Path:
    kb_dir = data_root / "upload" / name
    raw_dir = kb_dir / "raw"
    version_dir = kb_dir / "version-1"
    raw_dir.mkdir(parents=True)
    version_dir.mkdir()
    (raw_dir / "课本.pdf").write_bytes(b"%PDF-1.4\nlesson")
    (version_dir / "docstore.json").write_text(
        json.dumps({"docstore/data": {"node-1": {}}}), encoding="utf-8"
    )
    (version_dir / "index_store.json").write_text("{}", encoding="utf-8")
    (version_dir / "default__vector_store.json").write_text("{}", encoding="utf-8")
    (version_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": "version-1",
                "signature": "embedding-signature",
                "provider": "llamaindex",
                "model": "test-embedding",
                "dimension": 128,
            }
        ),
        encoding="utf-8",
    )
    (kb_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": name,
                "description": "Imported lessons",
                "rag_provider": "llamaindex",
                "created_at": "2026-08-01 10:00:00",
                "last_indexed_at": "2026-08-01 11:00:00",
                "last_indexed_count": 1,
                "last_indexed_action": "create",
            }
        ),
        encoding="utf-8",
    )
    (kb_dir / ".progress.json").write_text(
        json.dumps({"kb_name": name, "stage": "completed", "progress_percent": 100}),
        encoding="utf-8",
    )
    return kb_dir


def test_upload_browser_is_fixed_below_data_upload(tmp_path: Path) -> None:
    manager = KnowledgeBaseManager(base_dir=str(tmp_path / "data" / "knowledge_bases"))
    _write_ready_upload(tmp_path / "data")
    (tmp_path / "data" / "upload" / ".hidden").mkdir()
    (tmp_path / "data" / "upload" / "notes.txt").write_text("not a folder")

    listing = list_upload_folders(manager)

    assert listing.path == ""
    assert listing.parent is None
    assert [(item.name, item.path, item.candidate) for item in listing.folders] == [
        ("高中课程", "高中课程", True)
    ]
    with pytest.raises(ValueError, match="inside the upload directory"):
        resolve_upload_folder(manager, "../knowledge_bases")
    with pytest.raises(ValueError, match="inside the upload directory"):
        resolve_upload_folder(manager, str(tmp_path))


def test_query_only_admin_import_copies_and_registers_ready_kb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    source = _write_ready_upload(data_root)
    manager = KnowledgeBaseManager(base_dir=str(data_root / "knowledge_bases"))
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    probe = probe_import_folder(manager, "高中课程")
    result = import_existing_knowledge_base(manager, "高中课程")

    assert probe.ok is True
    assert probe.provider == "llamaindex"
    assert probe.document_count == 1
    assert result["status"] == "imported"
    assert source.is_dir(), "the upload source is retained after import"
    target = data_root / "knowledge_bases" / "高中课程"
    assert (target / "raw" / "课本.pdf").read_bytes().startswith(b"%PDF")

    config = json.loads((data_root / "knowledge_bases" / "kb_config.json").read_text())
    entry = config["knowledge_bases"]["高中课程"]
    assert entry["status"] == "ready"
    assert entry["rag_provider"] == "llamaindex"
    assert entry["imported_from"] == "高中课程"
    assert entry["embedding_model"] == "test-embedding"
    assert entry["embedding_dim"] == 128
    assert entry["index_versions"][0]["storage_path"] == str(target / "version-1")
    assert str(source) not in json.dumps(entry, ensure_ascii=False)


def test_import_can_publish_under_an_explicit_safe_name(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_ready_upload(data_root, "source-kb")
    manager = KnowledgeBaseManager(base_dir=str(data_root / "knowledge_bases"))

    import_existing_knowledge_base(manager, "source-kb", "课程副本")

    target = data_root / "knowledge_bases" / "课程副本"
    assert target.is_dir()
    metadata = json.loads((target / "metadata.json").read_text())
    progress = json.loads((target / ".progress.json").read_text())
    assert metadata["name"] == "课程副本"
    assert progress["kb_name"] == "课程副本"


def test_import_rejects_incomplete_or_live_kb(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source = _write_ready_upload(data_root)
    manager = KnowledgeBaseManager(base_dir=str(data_root / "knowledge_bases"))
    (source / "version-1" / "index_store.json").unlink()

    incomplete = probe_import_folder(manager, "高中课程")
    assert incomplete.ok is False
    assert "Incomplete" in (incomplete.error or "")

    (source / "version-1" / "index_store.json").write_text("{}", encoding="utf-8")
    (source / ".progress.json").write_text(
        json.dumps({"stage": "processing_file"}), encoding="utf-8"
    )
    live = probe_import_folder(manager, "高中课程")
    assert live.ok is False
    assert "still being built" in (live.error or "")


def test_import_rejects_symbolic_links(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source = _write_ready_upload(data_root)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (source / "raw" / "linked-secret").symlink_to(outside)
    manager = KnowledgeBaseManager(base_dir=str(data_root / "knowledge_bases"))

    probe = probe_import_folder(manager, "高中课程")

    assert probe.ok is False
    assert "symbolic links" in (probe.error or "")


def test_probe_rejects_linked_metadata_before_reading_it(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source = _write_ready_upload(data_root)
    outside = tmp_path / "outside-metadata.json"
    outside.write_text("not json", encoding="utf-8")
    (source / "metadata.json").unlink()
    (source / "metadata.json").symlink_to(outside)
    manager = KnowledgeBaseManager(base_dir=str(data_root / "knowledge_bases"))

    probe = probe_import_folder(manager, "高中课程")

    assert probe.ok is False
    assert "symbolic links" in (probe.error or "")
    assert "Invalid JSON" not in (probe.error or "")
