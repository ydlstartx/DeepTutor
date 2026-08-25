"""``resolve_kb_manifest`` is the one seam that reads a KB's document list.

Both consumers (the chat system-prompt inventory and the ``kb_files`` tool) go
through it, so per-user visibility has to hold here: a user's own KBs resolve,
an admin KB resolves only while it is granted, and anything else yields
``None`` rather than a listing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptutor.knowledge.manager import KnowledgeBaseManager
from deeptutor.multi_user.knowledge_access import (
    list_visible_knowledge_bases,
    resolve_kb,
    resolve_kb_manifest,
    resolve_kb_manifest_async,
)


def _make_kb(manager: KnowledgeBaseManager, name: str, *files: str) -> None:
    """Register a KB and stage documents into it, as an upload would."""
    raw = Path(manager.base_dir) / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for filename in files:
        (raw / filename).write_bytes(b"x" * 512)
    manager.register_knowledge_base(name, description=f"test KB {name}")


def test_user_sees_their_own_kb(mu_isolated_root, as_user) -> None:
    from deeptutor.multi_user.knowledge_access import current_kb_manager

    with as_user("u_alice", role="user"):
        _make_kb(current_kb_manager(), "alice-kb", "a.pdf", "b.pdf")

        manifest = resolve_kb_manifest("alice-kb")

        assert manifest is not None
        assert manifest.total == 2
        assert [document.name for document in manifest.documents] == ["a.pdf", "b.pdf"]


def test_pattern_and_limit_reach_the_filesystem(mu_isolated_root, as_user) -> None:
    from deeptutor.multi_user.knowledge_access import current_kb_manager

    with as_user("u_alice", role="user"):
        _make_kb(current_kb_manager(), "alice-kb", "a.pdf", "b.pdf", "notes.md")

        manifest = resolve_kb_manifest("alice-kb", pattern="*.pdf", limit=1)

        assert manifest is not None
        assert (manifest.total, manifest.matched, manifest.omitted) == (3, 2, 1)


def test_unknown_kb_yields_no_manifest(mu_isolated_root, as_user) -> None:
    with as_user("u_alice", role="user"):
        assert resolve_kb_manifest("does-not-exist") is None


def test_ungranted_admin_kb_yields_no_manifest(mu_isolated_root, as_user) -> None:
    """Naming an admin KB directly must not leak its file list (403 → None)."""
    from deeptutor.multi_user.knowledge_access import admin_kb_manager

    with as_user("u_admin", role="admin"):
        _make_kb(admin_kb_manager(), "admin-kb", "secret.pdf")

    with as_user("u_alice", role="user"):
        assert resolve_kb_manifest("admin:kb:admin-kb") is None


def test_empty_reference_yields_no_manifest(mu_isolated_root, as_user) -> None:
    with as_user("u_alice", role="user"):
        assert resolve_kb_manifest("") is None
        assert resolve_kb_manifest(None) is None


def test_query_only_user_sees_assigned_admin_kb_and_own_ima_only(
    mu_isolated_root, as_user, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from deeptutor.multi_user import knowledge_access
    from deeptutor.multi_user.knowledge_access import admin_kb_manager, current_kb_manager

    # Stage a legacy personal local KB before enabling the production policy;
    # query-only mode must not expose it merely because the directory exists.
    with as_user("u_alice", role="user"):
        _make_kb(current_kb_manager(), "legacy-local", "private.pdf")
        current_kb_manager().register_ima_kb("My IMA", "cid", "key", "ima-1")

    with as_user("u_admin", role="admin"):
        _make_kb(admin_kb_manager(), "assigned", "lesson.pdf")

    monkeypatch.setattr(
        knowledge_access,
        "load_grant",
        lambda _user_id: {
            "knowledge_bases": [{"resource_id": "admin:kb:assigned", "name": "assigned"}]
        },
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    with as_user("u_alice", role="user"):
        visible_ids = {item["id"] for item in list_visible_knowledge_bases()}
        assert visible_ids == {"user:kb:My IMA", "admin:kb:assigned"}
        assert resolve_kb("user:kb:My IMA").source == "user"
        assert resolve_kb("admin:kb:assigned").read_only is True
        with pytest.raises(HTTPException, match="administrator-assigned") as exc_info:
            resolve_kb("user:kb:legacy-local")
        assert exc_info.value.status_code == 403
        assert resolve_kb_manifest("legacy-local") is None


@pytest.mark.asyncio
async def test_ima_async_manifest_uses_remote_inventory(
    mu_isolated_root, as_user, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deeptutor.multi_user.knowledge_access import current_kb_manager

    class _RemoteClient:
        def __init__(self, _config) -> None:
            pass

        async def list_knowledge_tree(self, *, max_items: int) -> dict:
            assert max_items == 1000
            return {
                "items": [
                    {"type": "folder", "path": "books", "folder_id": "f1"},
                    {"type": "file", "path": "books/pmpp.pdf", "media_id": "m1"},
                ],
                "truncated": False,
            }

    monkeypatch.setattr(
        "deeptutor.services.rag.pipelines.ima.client.ImaClient",
        _RemoteClient,
    )

    with as_user("u_alice", role="user"):
        current_kb_manager().register_ima_kb("IMA", "cid", "key", "kb-1")

        manifest = await resolve_kb_manifest_async("IMA")

    assert manifest is not None
    assert manifest.enumerable
    assert manifest.total == 1
    assert [document.name for document in manifest.documents] == ["books/pmpp.pdf"]
    assert manifest.documents[0].size == -1


@pytest.mark.asyncio
async def test_assigned_admin_ima_manifest_uses_admin_credentials(
    mu_isolated_root, as_user, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deeptutor.multi_user import knowledge_access
    from deeptutor.multi_user.knowledge_access import admin_kb_manager
    from deeptutor.services.rag.pipelines.ima.config import save_account_settings

    captured = []

    class _RemoteClient:
        def __init__(self, config) -> None:
            captured.append(config)

        async def list_knowledge_tree(self, *, max_items: int) -> dict:
            return {"items": [], "truncated": False}

    monkeypatch.setattr(
        "deeptutor.services.rag.pipelines.ima.client.ImaClient",
        _RemoteClient,
    )
    monkeypatch.setattr(
        knowledge_access,
        "load_grant",
        lambda _user_id: {"knowledge_bases": [{"resource_id": "admin:kb:shared-ima"}]},
    )

    with as_user("u_admin", role="admin"):
        save_account_settings({"client_id": "admin-client", "api_key": "admin-key"})
        admin_kb_manager().register_ima_kb("shared-ima", "", "", "ima-library")

    with as_user("u_alice", role="user"):
        save_account_settings({"client_id": "alice-client", "api_key": "alice-key"})
        manifest = await resolve_kb_manifest_async("admin:kb:shared-ima")

    assert manifest is not None
    assert manifest.enumerable
    assert [(item.client_id, item.api_key) for item in captured] == [("admin-client", "admin-key")]
