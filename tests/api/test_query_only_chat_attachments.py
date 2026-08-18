from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deeptutor.api.routers import attachments as attachments_router
from deeptutor.services.storage import LocalDiskAttachmentStore


def test_query_only_does_not_block_chat_attachment_storage_or_preview(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DEEPTUTOR_KB_QUERY_ONLY", "true")
    store = LocalDiskAttachmentStore(tmp_path / "chat-attachments")
    url = asyncio.run(
        store.put(
            session_id="session",
            attachment_id="attachment",
            filename="notes.txt",
            data=b"temporary chat context",
            mime_type="text/plain",
        )
    )
    monkeypatch.setattr(attachments_router, "get_attachment_store", lambda: store)

    app = FastAPI()
    app.include_router(attachments_router.router, prefix="/api/attachments")
    with TestClient(app) as client:
        response = client.get(url)

    assert response.status_code == 200
    assert response.content == b"temporary chat context"
