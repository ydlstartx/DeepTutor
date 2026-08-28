"""GitHub sources are local-build inputs and stay inert on query-only servers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from deeptutor.knowledge.manager import KnowledgeBaseManager
from deeptutor.knowledge.policy import KnowledgeBaseWriteDisabledError
from deeptutor.services.github_source.sync import sync_source
from deeptutor.services.github_source.sync_service import GitHubSourceSyncService
from deeptutor.services.web_source.sync import sync_source as sync_web_source


class _ClientThatMustNotRun:
    async def get_latest_commit_sha(self, *_args, **_kwargs):
        raise AssertionError("query-only policy must reject before network access")


@pytest.mark.asyncio
async def test_query_only_sync_rejects_before_network_or_raw_write(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    base = tmp_path / "knowledge_bases"

    with pytest.raises(KnowledgeBaseWriteDisabledError):
        await sync_source(
            "published",
            {
                "id": "source-1",
                "repo": "owner/repo",
                "branch": "main",
                "path": "docs",
                "glob": "*.md",
            },
            base_dir=str(base),
            client=_ClientThatMustNotRun(),
        )

    assert not (base / "published" / "raw").exists()


@pytest.mark.asyncio
async def test_query_only_web_sync_rejects_before_network_or_raw_write(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    base = tmp_path / "knowledge_bases"
    monkeypatch.setattr(
        "deeptutor.services.web_source.sync.crawl_and_diff",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("query-only policy must reject before network access")
        ),
    )

    with pytest.raises(KnowledgeBaseWriteDisabledError):
        await sync_web_source(
            "published",
            {"id": "web-1", "url": "https://docs.example.com"},
            base_dir=str(base),
        )

    assert not (base / "published" / "raw").exists()


@pytest.mark.asyncio
async def test_query_only_background_cycle_does_not_scan_or_mutate(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    monkeypatch.setattr(
        "deeptutor.knowledge.manager.KnowledgeBaseManager",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("query-only background sync must stop before opening KB state")
        ),
    )

    await GitHubSourceSyncService(base_dir=str(tmp_path))._sync_one_cycle()


def test_query_only_manager_rejects_github_source_metadata_writes(monkeypatch, tmp_path) -> None:
    base = tmp_path / "knowledge_bases"
    manager = KnowledgeBaseManager(base_dir=str(base))
    (base / "published").mkdir(parents=True)
    manager.register_knowledge_base("published")
    monkeypatch.setattr(
        "deeptutor.multi_user.context.get_current_user",
        lambda: SimpleNamespace(is_admin=True),
    )
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")

    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.add_github_source("published", "owner/repo")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.remove_github_source("published", "source-1")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.update_github_source_state("published", "source-1", last_sync_status="error")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.add_web_source("published", "https://docs.example.com")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.remove_web_source("published", "web-1")
    with pytest.raises(KnowledgeBaseWriteDisabledError):
        manager.update_web_source_state("published", "web-1", last_sync_status="error")
