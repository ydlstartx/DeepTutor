"""Privacy-preserving activity aggregates for the multi-user admin surface.

The account store is a small JSON file and must not become a hot write path.
Activity therefore lives in its own SQLite database under ``data/system``.
Only timestamps and counters are stored here: never prompts, answers, session
titles, IP addresses, or user-agent strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

from . import paths
from .context import get_current_user_or_none

logger = logging.getLogger(__name__)

RETENTION_DAYS = 365
SEEN_WRITE_INTERVAL_SECONDS = 60.0
RECENT_ACTIVITY_SECONDS = 5 * 60
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_init_lock = threading.Lock()
_initialized_paths: set[Path] = set()
_seen_lock = threading.Lock()
_seen_cache: dict[tuple[Path, str], float] = {}


def _db_path() -> Path:
    # Resolve lazily so tests and embedded deployments can replace SYSTEM_ROOT.
    return paths.SYSTEM_ROOT / "activity" / "activity.db"


def _raw_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _ensure_schema() -> Path:
    path = _db_path()
    if path in _initialized_paths and path.exists():
        return path
    with _init_lock:
        if path in _initialized_paths and path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with _raw_connect(path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_activity (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL DEFAULT '',
                    last_login_at REAL,
                    last_seen_at REAL,
                    last_used_at REAL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_activity (
                    user_id TEXT NOT NULL,
                    day TEXT NOT NULL,
                    login_count INTEGER NOT NULL DEFAULT 0,
                    kb_queries INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, day)
                );

                CREATE TABLE IF NOT EXISTS turn_activity (
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
                    usage_reported INTEGER NOT NULL DEFAULT 0,
                    origin TEXT NOT NULL DEFAULT 'live'
                );

                CREATE INDEX IF NOT EXISTS idx_turn_activity_user_started
                    ON turn_activity(user_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS history_backfill (
                    user_id TEXT PRIMARY KEY,
                    source_signature TEXT NOT NULL DEFAULT '',
                    backfilled_at REAL NOT NULL
                );
                """
            )
            turn_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(turn_activity)").fetchall()
            }
            if "origin" not in turn_columns:
                conn.execute(
                    "ALTER TABLE turn_activity ADD COLUMN origin TEXT NOT NULL DEFAULT 'live'"
                )
        _initialized_paths.add(path)
    return path


def _connect() -> sqlite3.Connection:
    return _raw_connect(_ensure_schema())


def _local_day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().date().isoformat()


def _identity() -> tuple[str, str] | None:
    user = get_current_user_or_none()
    if user is None or not user.id:
        return None
    return user.id, user.username


def _upsert_timestamp(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    username: str,
    column: str,
    timestamp: float,
) -> None:
    if column not in {"last_login_at", "last_seen_at", "last_used_at"}:
        raise ValueError(f"Unsupported activity timestamp: {column}")
    conn.execute(
        f"""
        INSERT INTO user_activity (user_id, username, {column}, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username = excluded.username,
            {column} = CASE
                WHEN user_activity.{column} IS NULL
                  OR excluded.{column} > user_activity.{column}
                THEN excluded.{column}
                ELSE user_activity.{column}
            END,
            updated_at = MAX(user_activity.updated_at, excluded.updated_at)
        """,
        (user_id, username, timestamp, timestamp),
    )


def record_login(user_id: str, username: str, *, now: float | None = None) -> None:
    """Record one successful login without making authentication depend on it."""
    timestamp = float(now if now is not None else time.time())
    try:
        with _connect() as conn:
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_login_at",
                timestamp=timestamp,
            )
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_seen_at",
                timestamp=timestamp,
            )
            conn.execute(
                """
                INSERT INTO daily_activity (user_id, day, login_count)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, day) DO UPDATE SET
                    login_count = daily_activity.login_count + 1
                """,
                (user_id, _local_day(timestamp)),
            )
    except Exception:
        logger.warning("Could not record login activity for %s", user_id, exc_info=True)


def record_seen(*, now: float | None = None) -> None:
    """Record authenticated presence, throttled to one durable write per minute."""
    identity = _identity()
    if identity is None:
        return
    user_id, username = identity
    timestamp = float(now if now is not None else time.time())
    cache_key = (_db_path(), user_id)
    with _seen_lock:
        prior_seen = _seen_cache.get(cache_key, 0.0)
        if prior_seen > timestamp - SEEN_WRITE_INTERVAL_SECONDS:
            return
        _seen_cache[cache_key] = timestamp
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO user_activity (user_id, username, last_seen_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    last_seen_at = excluded.last_seen_at,
                    updated_at = excluded.updated_at
                WHERE user_activity.last_seen_at IS NULL
                   OR user_activity.last_seen_at <= ?
                """,
                (
                    user_id,
                    username,
                    timestamp,
                    timestamp,
                    timestamp - SEEN_WRITE_INTERVAL_SECONDS,
                ),
            )
    except Exception:
        logger.debug("Could not record presence for %s", user_id, exc_info=True)


def record_kb_query(*, now: float | None = None) -> None:
    """Count one access-checked knowledge-base query for the current user."""
    identity = _identity()
    if identity is None:
        return
    user_id, username = identity
    timestamp = float(now if now is not None else time.time())
    try:
        with _connect() as conn:
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_used_at",
                timestamp=timestamp,
            )
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_seen_at",
                timestamp=timestamp,
            )
            conn.execute(
                """
                INSERT INTO daily_activity (user_id, day, kb_queries)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id, day) DO UPDATE SET
                    kb_queries = daily_activity.kb_queries + 1
                """,
                (user_id, _local_day(timestamp)),
            )
    except Exception:
        logger.warning("Could not record KB activity for %s", user_id, exc_info=True)


def record_turn_started(
    turn_id: str,
    session_id: str,
    capability: str,
    *,
    now: float | None = None,
) -> None:
    """Record a meaningful user action and an idempotent turn row."""
    identity = _identity()
    if identity is None or not turn_id:
        return
    user_id, username = identity
    timestamp = float(now if now is not None else time.time())
    try:
        with _connect() as conn:
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_used_at",
                timestamp=timestamp,
            )
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_seen_at",
                timestamp=timestamp,
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO turn_activity (
                    turn_id, user_id, username, session_id, capability, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (turn_id, user_id, username, session_id, capability, timestamp),
            )
    except Exception:
        logger.warning("Could not record turn start %s", turn_id, exc_info=True)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def extract_usage_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the last standardized ``cost_summary`` in persisted turn events."""
    for event in reversed(events):
        if str(event.get("type") or "") != "result":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        candidates = (metadata.get("cost_summary"), metadata.get("metadata"))
        summary: Mapping[str, Any] | None = None
        if isinstance(candidates[0], Mapping):
            summary = candidates[0]
        elif isinstance(candidates[1], Mapping):
            nested = candidates[1].get("cost_summary")
            if isinstance(nested, Mapping):
                summary = nested
        if summary is not None:
            return {
                "total_calls": _non_negative_int(summary.get("total_calls")),
                "prompt_tokens": _non_negative_int(summary.get("prompt_tokens")),
                "completion_tokens": _non_negative_int(summary.get("completion_tokens")),
                "total_tokens": _non_negative_int(summary.get("total_tokens")),
                "total_cost_usd": _non_negative_float(summary.get("total_cost_usd")),
            }
    return None


def record_turn_finished(
    turn_id: str,
    session_id: str,
    capability: str,
    status: str,
    events: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
) -> None:
    """Finalize a turn row; repeated calls replace rather than double-count it."""
    identity = _identity()
    if not turn_id:
        return
    timestamp = float(now if now is not None else time.time())
    usage = extract_usage_summary(events)
    try:
        with _connect() as conn:
            existing = conn.execute(
                """
                SELECT user_id, username, session_id, capability
                FROM turn_activity
                WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
            if existing is not None:
                user_id = str(existing["user_id"] or "")
                username = str(existing["username"] or "")
                session_id = session_id or str(existing["session_id"] or "")
                capability = capability or str(existing["capability"] or "")
            elif identity is not None:
                user_id, username = identity
            else:
                # A turn normally has a start row. If it does not and the
                # request context is already gone, there is no safe owner to
                # invent; the canonical-history reconciliation will recover it.
                return
            _upsert_timestamp(
                conn,
                user_id=user_id,
                username=username,
                column="last_used_at",
                timestamp=timestamp,
            )
            conn.execute(
                """
                INSERT INTO turn_activity (
                    turn_id, user_id, username, session_id, capability,
                    started_at, finished_at, status, llm_calls,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, usage_reported, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live')
                ON CONFLICT(turn_id) DO UPDATE SET
                    finished_at = excluded.finished_at,
                    status = excluded.status,
                    llm_calls = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.llm_calls ELSE turn_activity.llm_calls END,
                    prompt_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.prompt_tokens ELSE turn_activity.prompt_tokens END,
                    completion_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.completion_tokens ELSE turn_activity.completion_tokens END,
                    total_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.total_tokens ELSE turn_activity.total_tokens END,
                    cost_usd = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.cost_usd ELSE turn_activity.cost_usd END,
                    usage_reported = MAX(
                        turn_activity.usage_reported, excluded.usage_reported
                    )
                """,
                (
                    turn_id,
                    user_id,
                    username,
                    session_id,
                    capability,
                    timestamp,
                    timestamp,
                    status,
                    (usage or {}).get("total_calls", 0),
                    (usage or {}).get("prompt_tokens", 0),
                    (usage or {}).get("completion_tokens", 0),
                    (usage or {}).get("total_tokens", 0),
                    (usage or {}).get("total_cost_usd", 0.0),
                    1 if usage is not None else 0,
                ),
            )
    except Exception:
        logger.warning("Could not finalize turn activity %s", turn_id, exc_info=True)


def delete_user_activity(user_id: str) -> None:
    if not user_id:
        return
    try:
        with _connect() as conn:
            for table in ("user_activity", "daily_activity", "turn_activity", "history_backfill"):
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
    except Exception:
        logger.warning("Could not remove activity for %s", user_id, exc_info=True)


def _source_signature(path: Path) -> str:
    signatures: list[str] = []
    for candidate in (path, Path(f"{path}-wal")):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        signatures.append(f"{candidate.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(signatures)


def _history_source_for_user(
    user: Mapping[str, Any],
    *,
    sole_admin_id: str | None,
) -> tuple[Path, str] | None:
    """Return the canonical SQLite history source when attribution is safe."""
    user_id = str(user.get("id") or "")
    if not _SAFE_USER_ID.fullmatch(user_id):
        return None
    if str(user.get("role") or "user") == "admin":
        # Every administrator currently shares data/user. Historical rows can
        # only be attributed to a person when exactly one admin account exists.
        if sole_admin_id != user_id:
            return None
        return paths.ADMIN_WORKSPACE_ROOT / "user" / "chat_history.db", "history_admin_shared"
    return paths.USERS_ROOT / user_id / "user" / "chat_history.db", "history_user"


def _cost_summaries_from_turn_events(
    conn: sqlite3.Connection,
    cutoff: float,
) -> dict[str, dict[str, Any]]:
    """Read one canonical result usage summary per turn without content."""
    summaries: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            """
            SELECT
                events.turn_id AS turn_id,
                COALESCE(
                    json_extract(events.metadata_json, '$.cost_summary.total_calls'),
                    json_extract(events.metadata_json, '$.metadata.cost_summary.total_calls')
                ) AS total_calls,
                COALESCE(
                    json_extract(events.metadata_json, '$.cost_summary.prompt_tokens'),
                    json_extract(events.metadata_json, '$.metadata.cost_summary.prompt_tokens')
                ) AS prompt_tokens,
                COALESCE(
                    json_extract(events.metadata_json, '$.cost_summary.completion_tokens'),
                    json_extract(events.metadata_json, '$.metadata.cost_summary.completion_tokens')
                ) AS completion_tokens,
                COALESCE(
                    json_extract(events.metadata_json, '$.cost_summary.total_tokens'),
                    json_extract(events.metadata_json, '$.metadata.cost_summary.total_tokens')
                ) AS total_tokens,
                COALESCE(
                    json_extract(events.metadata_json, '$.cost_summary.total_cost_usd'),
                    json_extract(events.metadata_json, '$.metadata.cost_summary.total_cost_usd')
                ) AS total_cost_usd
            FROM turns
            JOIN turn_events AS events ON events.turn_id = turns.id
            WHERE turns.created_at >= ?
              AND events.type = 'result'
              AND json_valid(events.metadata_json)
              AND (
                    json_type(events.metadata_json, '$.cost_summary') = 'object'
                 OR json_type(events.metadata_json, '$.metadata.cost_summary') = 'object'
              )
            ORDER BY events.seq
            """,
            (cutoff,),
        ).fetchall()
    except sqlite3.Error:
        return summaries
    for row in rows:
        turn_id = str(row["turn_id"] or "")
        if turn_id:
            summaries[turn_id] = {
                "total_calls": _non_negative_int(row["total_calls"]),
                "prompt_tokens": _non_negative_int(row["prompt_tokens"]),
                "completion_tokens": _non_negative_int(row["completion_tokens"]),
                "total_tokens": _non_negative_int(row["total_tokens"]),
                "total_cost_usd": _non_negative_float(row["total_cost_usd"]),
            }
    return summaries


def _cost_summaries_from_history(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Select numeric usage fields only; conversation payloads never leave SQLite."""
    summaries: dict[str, dict[str, Any]] = {}
    try:
        rows = conn.execute(
            """
            SELECT
                json_extract(event.value, '$.turn_id') AS turn_id,
                COALESCE(
                    json_extract(event.value, '$.metadata.cost_summary.total_calls'),
                    json_extract(event.value, '$.metadata.metadata.cost_summary.total_calls')
                ) AS total_calls,
                COALESCE(
                    json_extract(event.value, '$.metadata.cost_summary.prompt_tokens'),
                    json_extract(event.value, '$.metadata.metadata.cost_summary.prompt_tokens')
                ) AS prompt_tokens,
                COALESCE(
                    json_extract(event.value, '$.metadata.cost_summary.completion_tokens'),
                    json_extract(event.value, '$.metadata.metadata.cost_summary.completion_tokens')
                ) AS completion_tokens,
                COALESCE(
                    json_extract(event.value, '$.metadata.cost_summary.total_tokens'),
                    json_extract(event.value, '$.metadata.metadata.cost_summary.total_tokens')
                ) AS total_tokens,
                COALESCE(
                    json_extract(event.value, '$.metadata.cost_summary.total_cost_usd'),
                    json_extract(event.value, '$.metadata.metadata.cost_summary.total_cost_usd')
                ) AS total_cost_usd
            FROM messages,
                 json_each(
                    CASE WHEN json_valid(messages.events_json)
                         THEN messages.events_json ELSE '[]' END
                 ) AS event
            WHERE json_extract(event.value, '$.type') = 'result'
              AND (
                    json_type(event.value, '$.metadata.cost_summary') = 'object'
                 OR json_type(event.value, '$.metadata.metadata.cost_summary') = 'object'
              )
            """
        ).fetchall()
    except sqlite3.Error:
        return summaries
    for row in rows:
        turn_id = str(row["turn_id"] or "")
        if not turn_id:
            continue
        summaries[turn_id] = {
            "total_calls": _non_negative_int(row["total_calls"]),
            "prompt_tokens": _non_negative_int(row["prompt_tokens"]),
            "completion_tokens": _non_negative_int(row["completion_tokens"]),
            "total_tokens": _non_negative_int(row["total_tokens"]),
            "total_cost_usd": _non_negative_float(row["total_cost_usd"]),
        }
    return summaries


def _replace_turn_snapshot(
    user: Mapping[str, Any],
    turn_rows: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Mapping[str, Any]],
    *,
    origin: str,
    timestamp: float,
    signature: str | None = None,
    clear_origins: Sequence[str] = (),
) -> None:
    """Replace one canonical backend snapshot without losing live usage."""
    user_id = str(user.get("id") or "")
    username = str(user.get("username") or "")
    canonical_ids = {str(row.get("id") or "") for row in turn_rows}
    canonical_ids.discard("")
    latest = max(
        (
            float(row.get("finished_at") or row.get("updated_at") or row.get("created_at") or 0)
            for row in turn_rows
        ),
        default=0.0,
    )

    with _connect() as activity_conn:
        for stale_origin in clear_origins:
            if stale_origin != origin:
                activity_conn.execute(
                    "DELETE FROM turn_activity WHERE user_id = ? AND origin = ?",
                    (user_id, stale_origin),
                )

        # A live row is the freshest source while a request is running. Remove
        # only old live rows that the canonical store no longer knows about.
        stale_live = activity_conn.execute(
            """
            SELECT turn_id FROM turn_activity
            WHERE user_id = ? AND origin = 'live' AND started_at < ?
            """,
            (user_id, timestamp - RECENT_ACTIVITY_SECONDS),
        ).fetchall()
        for existing in stale_live:
            if str(existing["turn_id"] or "") not in canonical_ids:
                activity_conn.execute(
                    "DELETE FROM turn_activity WHERE turn_id = ?",
                    (existing["turn_id"],),
                )

        if latest > 0:
            _upsert_timestamp(
                activity_conn,
                user_id=user_id,
                username=username,
                column="last_used_at",
                timestamp=latest,
            )

        for row in turn_rows:
            turn_id = str(row.get("id") or "")
            if not turn_id:
                continue
            usage = costs.get(turn_id)
            finished_at = row.get("finished_at")
            activity_conn.execute(
                """
                INSERT INTO turn_activity (
                    turn_id, user_id, username, session_id, capability,
                    started_at, finished_at, status, llm_calls,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, usage_reported, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    username = excluded.username,
                    session_id = excluded.session_id,
                    capability = excluded.capability,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    status = excluded.status,
                    llm_calls = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.llm_calls ELSE turn_activity.llm_calls END,
                    prompt_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.prompt_tokens ELSE turn_activity.prompt_tokens END,
                    completion_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.completion_tokens ELSE turn_activity.completion_tokens END,
                    total_tokens = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.total_tokens ELSE turn_activity.total_tokens END,
                    cost_usd = CASE WHEN excluded.usage_reported = 1
                        THEN excluded.cost_usd ELSE turn_activity.cost_usd END,
                    usage_reported = MAX(
                        turn_activity.usage_reported, excluded.usage_reported
                    ),
                    origin = excluded.origin
                """,
                (
                    turn_id,
                    user_id,
                    username,
                    str(row.get("session_id") or ""),
                    str(row.get("capability") or ""),
                    float(row.get("created_at") or timestamp),
                    float(finished_at) if finished_at is not None else None,
                    str(row.get("status") or "completed"),
                    (usage or {}).get("total_calls", 0),
                    (usage or {}).get("prompt_tokens", 0),
                    (usage or {}).get("completion_tokens", 0),
                    (usage or {}).get("total_tokens", 0),
                    (usage or {}).get("total_cost_usd", 0.0),
                    1 if usage is not None else 0,
                    origin,
                ),
            )

        # Drop canonical rows that disappeared from the source. This happens
        # after upserts so a previously discovered usage summary survives a
        # briefly delayed/partial result-event read.
        old_snapshot = activity_conn.execute(
            "SELECT turn_id FROM turn_activity WHERE user_id = ? AND origin = ?",
            (user_id, origin),
        ).fetchall()
        for existing in old_snapshot:
            if str(existing["turn_id"] or "") not in canonical_ids:
                activity_conn.execute(
                    "DELETE FROM turn_activity WHERE turn_id = ?",
                    (existing["turn_id"],),
                )

        if signature is not None:
            activity_conn.execute(
                """
                INSERT INTO history_backfill (user_id, source_signature, backfilled_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    source_signature = excluded.source_signature,
                    backfilled_at = excluded.backfilled_at
                """,
                (user_id, signature, timestamp),
            )


def _record_value(record: Any, field: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def backfill_pocketbase_history(
    users: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
) -> dict[str, bool]:
    """Reconcile remote canonical turns using metadata-only PocketBase reads."""
    timestamp = float(now if now is not None else time.time())
    cutoff = timestamp - RETENTION_DAYS * 86400
    users_by_id = {
        str(user.get("id") or ""): user
        for user in users
        if _SAFE_USER_ID.fullmatch(str(user.get("id") or ""))
    }
    complete = {str(user.get("id") or ""): False for user in users}
    if not users_by_id:
        return complete

    try:
        from deeptutor.services.pocketbase_client import get_pb_client

        pb = get_pb_client()
        session_records = pb.collection("sessions").get_full_list(
            query_params={"fields": "session_id,user_id"}
        )
        turn_records = pb.collection("turns").get_full_list(
            query_params={
                "filter": f"turn_created_at >= {cutoff}",
                "fields": (
                    "turn_id,session_id,capability,status,turn_created_at,"
                    "turn_updated_at,finished_at"
                ),
            }
        )
    except Exception:
        logger.warning("Could not read PocketBase activity history", exc_info=True)
        return complete

    session_owners: dict[str, str] = {}
    unattributed_sessions: set[str] = set()
    ambiguous_sessions: set[str] = set()
    ambiguous_owners: set[str] = set()
    for record in session_records:
        session_id = str(_record_value(record, "session_id", "") or "")
        owner_id = str(_record_value(record, "user_id", "") or "")
        if not session_id:
            continue
        if not owner_id:
            unattributed_sessions.add(session_id)
            continue
        previous = session_owners.get(session_id)
        if previous is not None and previous != owner_id:
            ambiguous_sessions.add(session_id)
            if previous in users_by_id:
                ambiguous_owners.add(previous)
            if owner_id in users_by_id:
                ambiguous_owners.add(owner_id)
            continue
        session_owners[session_id] = owner_id

    rows_by_user: dict[str, list[dict[str, Any]]] = {user_id: [] for user_id in users_by_id}
    turn_owner: dict[str, str] = {}
    has_unattributed_turns = False
    for record in turn_records:
        session_id = str(_record_value(record, "session_id", "") or "")
        if session_id in unattributed_sessions:
            has_unattributed_turns = True
            continue
        if session_id in ambiguous_sessions:
            continue
        owner_id = session_owners.get(session_id)
        if owner_id not in users_by_id or owner_id in ambiguous_owners:
            continue
        turn_id = str(_record_value(record, "turn_id", "") or _record_value(record, "id", "") or "")
        if not turn_id:
            continue
        turn_owner[turn_id] = owner_id
        rows_by_user[owner_id].append(
            {
                "id": turn_id,
                "session_id": session_id,
                "capability": str(_record_value(record, "capability", "") or ""),
                "status": str(_record_value(record, "status", "running") or "running"),
                "created_at": _non_negative_float(
                    _record_value(record, "turn_created_at", timestamp)
                ),
                "updated_at": _non_negative_float(
                    _record_value(record, "turn_updated_at", timestamp)
                ),
                "finished_at": (
                    _non_negative_float(_record_value(record, "finished_at"))
                    if _record_value(record, "finished_at") not in (None, "")
                    else None
                ),
            }
        )

    costs: dict[str, dict[str, Any]] = {}
    try:
        event_records = pb.collection("turn_events").get_full_list(
            query_params={
                "filter": f'type="result" && event_timestamp >= {cutoff}',
                "sort": "seq",
                "fields": "turn_id,seq,type,metadata_json",
            }
        )
        for record in event_records:
            turn_id = str(_record_value(record, "turn_id", "") or "")
            if turn_id not in turn_owner:
                continue
            usage = extract_usage_summary(
                [
                    {
                        "type": "result",
                        "metadata": _json_mapping(_record_value(record, "metadata_json")),
                    }
                ]
            )
            if usage is not None:
                costs[turn_id] = usage
    except Exception:
        # Turn/status history is still exact. Missing usage is surfaced by
        # usage_complete=false instead of discarding the real turn counts.
        logger.warning("Could not read PocketBase usage events", exc_info=True)

    for user_id, user in users_by_id.items():
        if user_id in ambiguous_owners:
            continue
        _replace_turn_snapshot(
            user,
            rows_by_user[user_id],
            costs,
            origin="history_pocketbase",
            timestamp=timestamp,
            signature=f"history_pocketbase|synced:{timestamp}",
            clear_origins=("history_user", "history_admin_shared"),
        )
        complete[user_id] = not has_unattributed_turns
    return complete


def backfill_user_history(
    user: Mapping[str, Any],
    *,
    source: Path | None = None,
    origin: str = "history_user",
    now: float | None = None,
) -> bool:
    """Reconcile one user's aggregate cache from the canonical SQLite store."""
    user_id = str(user.get("id") or "")
    if not _SAFE_USER_ID.fullmatch(user_id):
        return False
    if source is None:
        source = paths.USERS_ROOT / user_id / "user" / "chat_history.db"
    allowed_root = (
        paths.ADMIN_WORKSPACE_ROOT if origin == "history_admin_shared" else paths.USERS_ROOT
    )
    try:
        if not source.resolve().is_relative_to(allowed_root.resolve()):
            return False
    except OSError:
        return False
    if not source.is_file():
        _replace_turn_snapshot(
            user,
            [],
            {},
            origin=origin,
            timestamp=float(now if now is not None else time.time()),
            clear_origins=("history_pocketbase",),
        )
        return True
    signature = f"{origin}|{_source_signature(source)}"
    if not signature:
        return False
    try:
        with _connect() as activity_conn:
            cached = activity_conn.execute(
                "SELECT source_signature FROM history_backfill WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if cached is not None and str(cached["source_signature"] or "") == signature:
            return True
    except Exception:
        # A cache read must never prevent rebuilding from the canonical store.
        logger.debug("Could not read history signature for %s", user_id, exc_info=True)

    try:
        uri = f"{source.resolve().as_uri()}?mode=ro"
        history_conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        history_conn.row_factory = sqlite3.Row
        try:
            tables = {
                str(row[0])
                for row in history_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "turns" not in tables:
                _replace_turn_snapshot(
                    user,
                    [],
                    {},
                    origin=origin,
                    timestamp=float(now if now is not None else time.time()),
                    signature=signature,
                    clear_origins=("history_pocketbase",),
                )
                return True
            cutoff = float(now if now is not None else time.time()) - RETENTION_DAYS * 86400
            turn_rows = history_conn.execute(
                """
                SELECT id, session_id, capability, status, created_at,
                       updated_at, finished_at
                FROM turns
                WHERE created_at >= ?
                """,
                (cutoff,),
            ).fetchall()
            costs = _cost_summaries_from_history(history_conn) if "messages" in tables else {}
            if "turn_events" in tables:
                # turn_events is the canonical per-turn trace and includes
                # results that may never have reached an assistant message.
                costs.update(_cost_summaries_from_turn_events(history_conn, cutoff))
        finally:
            history_conn.close()

        timestamp = float(now if now is not None else time.time())
        snapshot = [
            {
                "id": str(row["id"]),
                "session_id": str(row["session_id"] or ""),
                "capability": str(row["capability"] or ""),
                "status": str(row["status"] or "completed"),
                "created_at": float(row["created_at"] or timestamp),
                "updated_at": float(row["updated_at"] or timestamp),
                "finished_at": (
                    float(row["finished_at"]) if row["finished_at"] is not None else None
                ),
            }
            for row in turn_rows
        ]
        _replace_turn_snapshot(
            user,
            snapshot,
            costs,
            origin=origin,
            timestamp=timestamp,
            signature=signature,
            clear_origins=("history_pocketbase",),
        )
        return True
    except Exception:
        logger.warning("Could not backfill activity for %s", user_id, exc_info=True)
        return False


def _iso(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _first_included_day(cutoff: float) -> str:
    """First whole local calendar day after a rolling-window cutoff."""
    cutoff_day = datetime.fromtimestamp(cutoff).astimezone().date()
    return (cutoff_day + timedelta(days=1)).isoformat()


def _usage_for(
    conn: sqlite3.Connection,
    user_id: str,
    cutoff: float,
    *,
    history_complete: bool,
) -> dict[str, Any]:
    turns = conn.execute(
        """
        SELECT
            COUNT(*) AS turns,
            COUNT(DISTINCT session_id) AS conversations,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN status IN ('failed', 'rejected') THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running,
            SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
            SUM(llm_calls) AS llm_calls,
            SUM(prompt_tokens) AS prompt_tokens,
            SUM(completion_tokens) AS completion_tokens,
            SUM(total_tokens) AS total_tokens,
            SUM(cost_usd) AS cost_usd,
            SUM(CASE WHEN status IN ('completed', 'failed', 'cancelled') THEN 1 ELSE 0 END)
                AS usage_expected,
            SUM(CASE
                    WHEN status IN ('completed', 'failed', 'cancelled')
                     AND usage_reported = 1
                    THEN 1 ELSE 0
                END) AS usage_reported
        FROM turn_activity
        WHERE user_id = ? AND started_at >= ?
        """,
        (user_id, cutoff),
    ).fetchone()
    kb = conn.execute(
        "SELECT COALESCE(SUM(kb_queries), 0) FROM daily_activity WHERE user_id = ? AND day >= ?",
        (user_id, _first_included_day(cutoff)),
    ).fetchone()[0]
    completed = _non_negative_int(turns["completed"])
    expected = _non_negative_int(turns["usage_expected"])
    reported = _non_negative_int(turns["usage_reported"])
    return {
        "conversations": _non_negative_int(turns["conversations"]),
        "turns": _non_negative_int(turns["turns"]),
        "completed_turns": completed,
        "failed_turns": _non_negative_int(turns["failed"]),
        "running_turns": _non_negative_int(turns["running"]),
        "cancelled_turns": _non_negative_int(turns["cancelled"]),
        "kb_queries": _non_negative_int(kb),
        "llm_calls": _non_negative_int(turns["llm_calls"]),
        "prompt_tokens": _non_negative_int(turns["prompt_tokens"]),
        "completion_tokens": _non_negative_int(turns["completion_tokens"]),
        "total_tokens": _non_negative_int(turns["total_tokens"]),
        "estimated_cost_usd": round(_non_negative_float(turns["cost_usd"]), 6),
        "usage_complete": history_complete and expected == reported,
        "usage_reported_turns": reported,
        "history_complete": history_complete,
    }


def get_activity_report(
    users: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Build the admin report and opportunistically backfill isolated history."""
    timestamp = float(now if now is not None else time.time())
    admin_ids = [
        str(user.get("id") or "") for user in users if str(user.get("role") or "user") == "admin"
    ]
    sole_admin_id = admin_ids[0] if len(admin_ids) == 1 else None
    history_complete_by_user: dict[str, bool] = {}

    try:
        from deeptutor.services.pocketbase_client import is_pocketbase_enabled

        pocketbase_enabled = is_pocketbase_enabled()
    except Exception:
        pocketbase_enabled = False

    if len(admin_ids) != 1:
        # Shared admin history cannot safely remain assigned after a second
        # admin is introduced.
        with _connect() as conn:
            conn.execute("DELETE FROM turn_activity WHERE origin = 'history_admin_shared'")
            conn.execute(
                "DELETE FROM history_backfill WHERE source_signature LIKE 'history_admin_shared|%'"
            )

    if pocketbase_enabled:
        history_complete_by_user = backfill_pocketbase_history(users, now=timestamp)
    else:
        for user in users:
            user_id = str(user.get("id") or "")
            source_info = _history_source_for_user(user, sole_admin_id=sole_admin_id)
            if source_info is None:
                history_complete_by_user[user_id] = False
                continue
            source, origin = source_info
            history_complete_by_user[user_id] = backfill_user_history(
                user,
                source=source,
                origin=origin,
                now=timestamp,
            )

    cutoff_7d = timestamp - 7 * 86400
    cutoff_30d = timestamp - 30 * 86400
    retention_cutoff = timestamp - RETENTION_DAYS * 86400
    local_now = datetime.fromtimestamp(timestamp).astimezone()
    start_today = local_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    results: list[dict[str, Any]] = []
    activity_markers: list[float] = []

    with _connect() as conn:
        conn.execute("DELETE FROM turn_activity WHERE started_at < ?", (retention_cutoff,))
        conn.execute("DELETE FROM daily_activity WHERE day < ?", (_local_day(retention_cutoff),))
        activity_rows = {
            str(row["user_id"]): row
            for row in conn.execute("SELECT * FROM user_activity").fetchall()
        }
        for user in users:
            user_id = str(user.get("id") or "")
            activity = activity_rows.get(user_id)
            last_login = activity["last_login_at"] if activity is not None else None
            last_seen = activity["last_seen_at"] if activity is not None else None
            last_used = activity["last_used_at"] if activity is not None else None
            marker = max(
                (float(value) for value in (last_seen, last_used, last_login) if value is not None),
                default=0.0,
            )
            activity_markers.append(marker)
            if marker >= timestamp - RECENT_ACTIVITY_SECONDS:
                status = "recent"
            elif marker >= start_today:
                status = "today"
            elif marker >= cutoff_7d:
                status = "recent_7d"
            else:
                status = "inactive"
            results.append(
                {
                    "id": user_id,
                    "username": str(user.get("username") or ""),
                    "role": str(user.get("role") or "user"),
                    "created_at": str(user.get("created_at") or ""),
                    "disabled": bool(user.get("disabled", False)),
                    "avatar": str(user.get("avatar") or ""),
                    "activity_status": status,
                    "last_activity_at": _iso(marker) if marker > 0 else None,
                    "last_login_at": _iso(last_login),
                    "last_seen_at": _iso(last_seen),
                    "last_used_at": _iso(last_used),
                    "usage_7d": _usage_for(
                        conn,
                        user_id,
                        cutoff_7d,
                        history_complete=history_complete_by_user.get(user_id, False),
                    ),
                    "usage_30d": _usage_for(
                        conn,
                        user_id,
                        cutoff_30d,
                        history_complete=history_complete_by_user.get(user_id, False),
                    ),
                }
            )

    active_today = sum(marker >= start_today for marker in activity_markers)
    active_7d = sum(marker >= cutoff_7d for marker in activity_markers)
    inactive_30d = sum(marker < cutoff_30d for marker in activity_markers)
    return {
        "summary": {
            "total_users": len(results),
            "active_today": active_today,
            "active_7d": active_7d,
            "inactive_30d": inactive_30d,
        },
        "users": results,
        "generated_at": _iso(timestamp),
        "retention_days": RETENTION_DAYS,
    }


__all__ = [
    "backfill_pocketbase_history",
    "backfill_user_history",
    "delete_user_activity",
    "extract_usage_summary",
    "get_activity_report",
    "record_kb_query",
    "record_login",
    "record_seen",
    "record_turn_finished",
    "record_turn_started",
]
