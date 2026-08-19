from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from deeptutor.knowledge.manager import KnowledgeBaseManager
from deeptutor.knowledge.policy import KnowledgeBaseWriteDisabledError


def test_query_only_list_does_not_rewrite_or_prune_kb_config(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    ready = base / "published" / "version-1"
    ready.mkdir(parents=True)
    (ready / "docstore.json").write_text("{}", encoding="utf-8")
    (ready / "index_store.json").write_text("{}", encoding="utf-8")
    (ready / "meta.json").write_text(
        json.dumps({"provider": "llamaindex", "signature": "published"}),
        encoding="utf-8",
    )
    config_path = base / "kb_config.json"
    original = json.dumps(
        {
            "default": "orphan",
            "knowledge_bases": {"orphan": {"path": "missing", "rag_provider": "removed-provider"}},
        }
    )
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    manager = KnowledgeBaseManager(base_dir=str(base))
    assert manager.list_knowledge_bases() == ["published"]
    assert config_path.read_text(encoding="utf-8") == original


def test_query_only_delete_is_rejected_before_files_are_touched(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    kb_dir = base / "published"
    kb_dir.mkdir(parents=True)
    marker = kb_dir / "index.dat"
    marker.write_bytes(b"published index")
    (base / "kb_config.json").write_text(
        json.dumps({"knowledge_bases": {"published": {"path": "published"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    monkeypatch.setattr(
        "deeptutor.multi_user.context.get_current_user",
        lambda: SimpleNamespace(is_admin=False),
    )

    manager = KnowledgeBaseManager(base_dir=str(base))
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.delete_knowledge_base("published", confirm=True)

    assert marker.read_bytes() == b"published index"


def test_query_only_admin_can_delete_published_kb(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    kb_dir = base / "published"
    kb_dir.mkdir(parents=True)
    (kb_dir / "index.dat").write_bytes(b"published index")
    (base / "kb_config.json").write_text(
        json.dumps(
            {
                "default": "published",
                "knowledge_bases": {"published": {"path": "published"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    manager = KnowledgeBaseManager(base_dir=str(base))
    assert manager.delete_knowledge_base("published", confirm=True) is True

    assert not kb_dir.exists()
    persisted = json.loads((base / "kb_config.json").read_text(encoding="utf-8"))
    assert "published" not in persisted["knowledge_bases"]
    assert persisted["default"] is None


def test_query_only_admin_delete_ima_only_drops_pointer(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    stale_local_dir = base / "IMA"
    stale_local_dir.mkdir(parents=True)
    marker = stale_local_dir / "must-not-be-deleted"
    marker.write_text("remote content is not managed here", encoding="utf-8")
    (base / "kb_config.json").write_text(
        json.dumps(
            {
                "knowledge_bases": {
                    "IMA": {
                        "path": "IMA",
                        "type": "ima",
                        "rag_provider": "ima",
                        "knowledge_base_id": "remote-kb",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    manager = KnowledgeBaseManager(base_dir=str(base))
    assert manager.delete_knowledge_base("IMA", confirm=True) is True

    assert marker.read_text(encoding="utf-8") == "remote content is not managed here"
    persisted = json.loads((base / "kb_config.json").read_text(encoding="utf-8"))
    assert "IMA" not in persisted["knowledge_bases"]


def test_query_only_allows_only_ima_pointer_registration(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    base.mkdir()
    original_entry = {"path": "published", "rag_provider": "removed-provider"}
    (base / "kb_config.json").write_text(
        json.dumps(
            {
                "default": "published",
                "knowledge_bases": {"published": original_entry},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    manager = KnowledgeBaseManager(base_dir=str(base))

    entry = manager.register_ima_kb("IMA", "", "", "remote-kb")

    assert entry["type"] == "ima"
    assert entry["knowledge_base_id"] == "remote-kb"
    assert not (base / "IMA").exists()
    persisted = json.loads((base / "kb_config.json").read_text(encoding="utf-8"))
    assert persisted["knowledge_bases"]["IMA"]["type"] == "ima"
    assert persisted["knowledge_bases"]["published"] == original_entry
    assert persisted["default"] == "published"

    local_dir = base / "local"
    local_dir.mkdir()
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.register_knowledge_base("local")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.register_lightrag_server_kb("remote", "https://example.invalid")
