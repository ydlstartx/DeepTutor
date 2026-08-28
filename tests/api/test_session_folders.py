from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from deeptutor.services.session.sqlite_store import SQLiteSessionStore


def test_session_folder_api_roundtrip(monkeypatch, tmp_path: Path) -> None:
    sessions_module = importlib.import_module("deeptutor.api.routers.sessions")
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    session = asyncio.run(store.create_session(title="Folder API"))
    monkeypatch.setattr(sessions_module, "get_session_store", lambda: store)

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1/sessions")

    with TestClient(app) as client:
        created = client.post("/api/v1/sessions/folders", json={"name": "Work"})
        assert created.status_code == 201
        folder = created.json()["folder"]

        moved = client.put(
            f"/api/v1/sessions/{session['id']}/folder",
            json={"folder_id": folder["id"]},
        )
        assert moved.status_code == 200

        listed = client.get("/api/v1/sessions").json()["sessions"]
        assert listed[0]["folder_id"] == folder["id"]
        assert client.get("/api/v1/sessions/folders").json()["folders"][0]["session_count"] == 1

        duplicate = client.post("/api/v1/sessions/folders", json={"name": " work "})
        assert duplicate.status_code == 409

        deleted = client.delete(f"/api/v1/sessions/folders/{folder['id']}")
        assert deleted.status_code == 200
        listed = client.get("/api/v1/sessions").json()["sessions"]
        assert listed[0]["folder_id"] is None


def test_session_folder_api_rejects_unknown_folder(monkeypatch, tmp_path: Path) -> None:
    sessions_module = importlib.import_module("deeptutor.api.routers.sessions")
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    session = asyncio.run(store.create_session())
    monkeypatch.setattr(sessions_module, "get_session_store", lambda: store)

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1/sessions")
    with TestClient(app) as client:
        response = client.put(
            f"/api/v1/sessions/{session['id']}/folder",
            json={"folder_id": "folder_missing"},
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "Folder not found"


def test_moving_parent_chat_keeps_little_tutor_in_same_folder(monkeypatch, tmp_path: Path) -> None:
    sessions_module = importlib.import_module("deeptutor.api.routers.sessions")
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    parent = asyncio.run(store.create_session(title="Parent"))
    child = asyncio.run(store.create_session(title="Little Tutor"))
    asyncio.run(
        store.update_session_preferences(
            child["id"],
            {
                "parent_session_id": parent["id"],
                "session_kind": "selection_tutor",
            },
        )
    )
    folder = asyncio.run(store.create_session_folder("Course notes"))
    monkeypatch.setattr(sessions_module, "get_session_store", lambda: store)

    app = FastAPI()
    app.include_router(sessions_module.router, prefix="/api/v1/sessions")
    with TestClient(app) as client:
        response = client.put(
            f"/api/v1/sessions/{parent['id']}/folder",
            json={"folder_id": folder["id"]},
        )

    assert response.status_code == 200
    assert asyncio.run(store.get_session(parent["id"]))["folder_id"] == folder["id"]
    assert asyncio.run(store.get_session(child["id"]))["folder_id"] == folder["id"]
