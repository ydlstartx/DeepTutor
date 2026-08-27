"""Admin activity aggregates stay scoped, private, and idempotent."""

from __future__ import annotations

import json
import sqlite3
import time


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
    assert item["last_login_at"] is not None
    assert item["last_seen_at"] is not None
    assert item["last_used_at"] is not None
    assert item["usage_7d"] == {
        "conversations": 1,
        "turns": 1,
        "completed_turns": 1,
        "failed_turns": 0,
        "kb_queries": 1,
        "llm_calls": 2,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "estimated_cost_usd": 0.0125,
        "usage_complete": True,
        "usage_reported_turns": 1,
    }
    serialized = json.dumps(report)
    assert "private answer" not in serialized
    assert "session-1" not in serialized


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
