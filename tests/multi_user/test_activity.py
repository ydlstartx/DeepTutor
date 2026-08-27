"""Admin activity aggregates stay scoped, private, and idempotent."""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace


def _users() -> list[dict]:
    return [
        {
            "id": "u_alice",
            "username": "alice",
            "role": "user",
            "created_at": "2026-01-01T00:00:00+00:00",
            "disabled": False,
            "avatar": "",
        }
    ]


def _cost_event(turn_id: str = "turn-1") -> dict:
    return {
        "type": "result",
        "turn_id": turn_id,
        "metadata": {
            "response": "private answer",
            "metadata": {
                "cost_summary": {
                    "total_calls": 2,
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "total_cost_usd": 0.0125,
                }
            },
        },
    }


def test_activity_report_tracks_presence_meaningful_use_and_usage(mu_isolated_root, as_user):
    from deeptutor.multi_user.activity import (
        get_activity_report,
        record_kb_query,
        record_login,
        record_seen,
        record_turn_finished,
        record_turn_started,
    )

    now = time.time()
    record_login("u_alice", "alice", now=now - 120)
    with as_user("u_alice", username="alice"):
        record_seen(now=now - 90)
        # Throttled: this request is less than a minute after the durable one.
        record_seen(now=now - 60)
        record_turn_started("turn-1", "session-1", "chat", now=now - 50)
        record_turn_finished(
            "turn-1",
            "session-1",
            "chat",
            "completed",
            [_cost_event()],
            now=now - 40,
        )
        record_kb_query(now=now - 30)

    report = get_activity_report(_users(), now=now)
    assert report["summary"] == {
        "total_users": 1,
        "active_today": 1,
        "active_7d": 1,
        "inactive_30d": 0,
    }
    item = report["users"][0]
    assert item["activity_status"] == "recent"
    assert item["last_activity_at"] is not None
    assert item["last_login_at"] is not None
    assert item["last_seen_at"] is not None
    assert item["last_used_at"] is not None
    assert item["usage_7d"] == {
        "conversations": 1,
        "turns": 1,
        "completed_turns": 1,
        "failed_turns": 0,
        "running_turns": 0,
        "cancelled_turns": 0,
        "kb_queries": 1,
        "llm_calls": 2,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "estimated_cost_usd": 0.0125,
        "usage_complete": True,
        "usage_reported_turns": 1,
        "history_complete": True,
    }
    serialized = json.dumps(report)
    assert "private answer" not in serialized
    assert "session-1" not in serialized


def test_login_only_user_counts_as_active_without_meaningful_use(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report, record_login

    now = time.time()
    record_login("u_alice", "alice", now=now - 30)

    report = get_activity_report(_users(), now=now)
    item = report["users"][0]
    assert report["summary"] == {
        "total_users": 1,
        "active_today": 1,
        "active_7d": 1,
        "inactive_30d": 0,
    }
    assert item["activity_status"] == "recent"
    assert item["last_activity_at"] == item["last_login_at"]
    assert item["last_used_at"] is None


def test_existing_activity_database_adds_snapshot_origin_column():
    from deeptutor.multi_user import activity

    db = activity._db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE turn_activity (
                turn_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                capability TEXT NOT NULL DEFAULT '',
                started_at REAL NOT NULL,
                finished_at REAL,
                status TEXT NOT NULL DEFAULT 'running',
                llm_calls INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                usage_reported INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    assert activity._ensure_schema() == db

    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(turn_activity)")}
    assert "origin" in columns


def test_repeated_turn_finalization_replaces_instead_of_double_counting(mu_isolated_root, as_user):
    from deeptutor.multi_user.activity import (
        get_activity_report,
        record_turn_finished,
        record_turn_started,
    )

    now = time.time()
    with as_user("u_alice", username="alice"):
        record_turn_started("turn-1", "session-1", "chat", now=now - 10)
        record_turn_finished("turn-1", "session-1", "chat", "failed", [], now=now - 5)
        record_turn_finished(
            "turn-1",
            "session-1",
            "chat",
            "completed",
            [_cost_event()],
            now=now,
        )

    usage = get_activity_report(_users(), now=now)["users"][0]["usage_7d"]
    assert usage["turns"] == 1
    assert usage["completed_turns"] == 1
    assert usage["failed_turns"] == 0
    assert usage["llm_calls"] == 2


def test_turn_finalization_uses_the_existing_owner_after_context_is_gone(
    mu_isolated_root,
    as_user,
):
    from deeptutor.multi_user.activity import (
        get_activity_report,
        record_turn_finished,
        record_turn_started,
    )

    now = time.time()
    with as_user("u_alice", username="alice"):
        record_turn_started("turn-context", "session-1", "chat", now=now - 5)

    record_turn_finished(
        "turn-context",
        "session-1",
        "chat",
        "completed",
        [_cost_event("turn-context")],
        now=now,
    )

    usage = get_activity_report(_users(), now=now)["users"][0]["usage_7d"]
    assert usage["completed_turns"] == 1
    assert usage["running_turns"] == 0
    assert usage["total_tokens"] == 150


def test_failed_turn_without_usage_summary_is_marked_partial(mu_isolated_root, as_user):
    from deeptutor.multi_user.activity import (
        get_activity_report,
        record_turn_finished,
        record_turn_started,
    )

    now = time.time()
    with as_user("u_alice", username="alice"):
        record_turn_started("turn-failed", "session-1", "chat", now=now - 5)
        record_turn_finished(
            "turn-failed",
            "session-1",
            "chat",
            "failed",
            [],
            now=now,
        )

    usage = get_activity_report(_users(), now=now)["users"][0]["usage_7d"]
    assert usage["failed_turns"] == 1
    assert usage["usage_reported_turns"] == 0
    assert usage["usage_complete"] is False


def test_existing_user_history_is_backfilled_without_content(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report

    now = time.time()
    db = mu_isolated_root / "data" / "users" / "u_alice" / "user" / "chat_history.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, title TEXT, created_at REAL, updated_at REAL
            );
            CREATE TABLE turns (
                id TEXT PRIMARY KEY, session_id TEXT, capability TEXT, status TEXT,
                created_at REAL, updated_at REAL, finished_at REAL
            );
            CREATE TABLE messages (id INTEGER PRIMARY KEY, events_json TEXT);
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            ("history-session", "Secret title", now - 200, now - 100),
        )
        conn.execute(
            "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "history-turn",
                "history-session",
                "chat",
                "completed",
                now - 180,
                now - 120,
                now - 120,
            ),
        )
        conn.execute(
            "INSERT INTO messages(events_json) VALUES (?)",
            (json.dumps([_cost_event("history-turn")]),),
        )

    report = get_activity_report(_users(), now=now)
    item = report["users"][0]
    assert item["usage_7d"]["conversations"] == 1
    assert item["usage_7d"]["completed_turns"] == 1
    assert item["usage_7d"]["total_tokens"] == 150
    assert "Secret title" not in json.dumps(report)


def test_history_usage_is_reconciled_after_the_first_scan(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report

    now = time.time()
    db = mu_isolated_root / "data" / "users" / "u_alice" / "user" / "chat_history.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE turns (
                id TEXT PRIMARY KEY, session_id TEXT, capability TEXT, status TEXT,
                created_at REAL, updated_at REAL, finished_at REAL
            );
            CREATE TABLE messages (id INTEGER PRIMARY KEY, events_json TEXT);
            """
        )
        conn.execute(
            "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("later-cost", "session-1", "chat", "completed", now - 20, now - 10, now - 10),
        )

    first = get_activity_report(_users(), now=now)["users"][0]["usage_7d"]
    assert first["completed_turns"] == 1
    assert first["total_tokens"] == 0
    assert first["usage_complete"] is False

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO messages(events_json) VALUES (?)",
            (json.dumps([_cost_event("later-cost")]),),
        )

    second = get_activity_report(_users(), now=now)["users"][0]["usage_7d"]
    assert second["total_tokens"] == 150
    assert second["usage_complete"] is True


def test_sole_admin_history_is_safely_attributed(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report

    now = time.time()
    db = mu_isolated_root / "data" / "user" / "chat_history.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE turns (
                id TEXT PRIMARY KEY, session_id TEXT, capability TEXT, status TEXT,
                created_at REAL, updated_at REAL, finished_at REAL
            );
            CREATE TABLE messages (id INTEGER PRIMARY KEY, events_json TEXT);
            """
        )
        conn.execute(
            "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("admin-turn", "admin-session", "chat", "completed", now - 20, now - 10, now - 10),
        )
        conn.execute(
            "INSERT INTO messages(events_json) VALUES (?)",
            (json.dumps([_cost_event("admin-turn")]),),
        )

    admin = {**_users()[0], "id": "admin-1", "username": "admin", "role": "admin"}
    usage = get_activity_report([admin], now=now)["users"][0]["usage_7d"]
    assert usage["conversations"] == 1
    assert usage["completed_turns"] == 1
    assert usage["total_tokens"] == 150
    assert usage["history_complete"] is True


def test_shared_admin_history_is_not_falsely_split_between_admins(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report

    admins = [
        {**_users()[0], "id": "admin-1", "username": "one", "role": "admin"},
        {**_users()[0], "id": "admin-2", "username": "two", "role": "admin"},
    ]
    report = get_activity_report(admins, now=time.time())

    assert all(item["usage_7d"]["turns"] == 0 for item in report["users"])
    assert all(item["usage_7d"]["history_complete"] is False for item in report["users"])


def test_history_backfill_rejects_user_id_path_traversal(mu_isolated_root):
    from deeptutor.multi_user.activity import get_activity_report

    now = time.time()
    outside = mu_isolated_root / "data" / "outside" / "user" / "chat_history.db"
    outside.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(outside) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, updated_at REAL)")
        conn.execute("INSERT INTO sessions VALUES (?, ?)", ("private", now))

    user = {**_users()[0], "id": "../outside"}
    item = get_activity_report([user], now=now)["users"][0]
    assert item["last_used_at"] is None


def test_delete_user_activity_removes_summary_and_counters(mu_isolated_root, as_user):
    from deeptutor.multi_user.activity import (
        delete_user_activity,
        get_activity_report,
        record_turn_started,
    )

    now = time.time()
    with as_user("u_alice", username="alice"):
        record_turn_started("turn-1", "session-1", "chat", now=now)
    delete_user_activity("u_alice")

    item = get_activity_report(_users(), now=now)["users"][0]
    assert item["last_used_at"] is None
    assert item["usage_7d"]["turns"] == 0


def test_pocketbase_history_is_reconciled_by_session_owner(
    mu_isolated_root,
    monkeypatch,
):
    from deeptutor.multi_user.activity import get_activity_report

    now = time.time()

    class Collection:
        def __init__(self, rows):
            self.rows = rows
            self.queries = []

        def get_full_list(self, query_params=None):
            self.queries.append(query_params or {})
            return self.rows

    class Client:
        def __init__(self):
            self.collections = {
                "sessions": Collection(
                    [
                        SimpleNamespace(session_id="alice-session", user_id="u_alice"),
                        SimpleNamespace(session_id="other-session", user_id="u_other"),
                    ]
                ),
                "turns": Collection(
                    [
                        SimpleNamespace(
                            turn_id="alice-turn",
                            session_id="alice-session",
                            capability="chat",
                            status="completed",
                            turn_created_at=now - 30,
                            turn_updated_at=now - 20,
                            finished_at=now - 20,
                        ),
                        SimpleNamespace(
                            turn_id="other-turn",
                            session_id="other-session",
                            capability="chat",
                            status="completed",
                            turn_created_at=now - 30,
                            turn_updated_at=now - 20,
                            finished_at=now - 20,
                        ),
                    ]
                ),
                "turn_events": Collection(
                    [
                        SimpleNamespace(
                            turn_id="alice-turn",
                            seq=10,
                            type="result",
                            metadata_json=_cost_event("alice-turn")["metadata"],
                        )
                    ]
                ),
            }
            self.accessed = []

        def collection(self, name):
            self.accessed.append(name)
            return self.collections[name]

    client = Client()
    monkeypatch.setattr(
        "deeptutor.services.pocketbase_client.is_pocketbase_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "deeptutor.services.pocketbase_client.get_pb_client",
        lambda: client,
    )

    report = get_activity_report(_users(), now=now)
    usage = report["users"][0]["usage_7d"]
    assert usage["conversations"] == 1
    assert usage["completed_turns"] == 1
    assert usage["total_tokens"] == 150
    assert usage["history_complete"] is True
    assert usage["usage_complete"] is True
    assert "messages" not in client.accessed
    assert "private answer" not in json.dumps(report)
    assert all(
        "fields" in query
        for collection in client.collections.values()
        for query in collection.queries
    )
