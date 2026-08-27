"""Privacy-preserving activity aggregates for the multi-user admin surface.

The account store is a small JSON file and must not become a hot write path.
Activity therefore lives in its own SQLite database under ``data/system``.
Only timestamps and counters are stored here: never prompts, answers, session
titles, IP addresses, or user-agent strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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
                    usage_reported INTEGER NOT NULL DEFAULT 0
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
    if identity is None or not turn_id:
        return
    user_id, username = identity
    timestamp = float(now if now is not None else time.time())
    usage = extract_usage_summary(events)
    try:
        with _connect() as conn:
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
                    cost_usd, usage_reported
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    finished_at = excluded.finished_at,
                    status = excluded.status,
                    llm_calls = excluded.llm_calls,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    cost_usd = excluded.cost_usd,
                    usage_reported = excluded.usage_reported
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


def _history_db_for_user(user_id: str) -> Path:
    # Never inspect the shared admin DB: multiple administrators use the same
    # workspace, so historical rows cannot be attributed to one account.
    return paths.USERS_ROOT / user_id / "user" / "chat_history.db"


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


def backfill_user_history(user: Mapping[str, Any], *, now: float | None = None) -> None:
    """Import non-sensitive turn metadata from one isolated user database."""
    user_id = str(user.get("id") or "")
    username = str(user.get("username") or "")
    if not _SAFE_USER_ID.fullmatch(user_id):
        return
    source = _history_db_for_user(user_id)
    try:
        if not source.resolve().is_relative_to(paths.USERS_ROOT.resolve()):
            return
    except OSError:
        return
    if not source.is_file():
        return
    signature = _source_signature(source)
    if not signature:
        return

    try:
        with _connect() as activity_conn:
            prior = activity_conn.execute(
                "SELECT source_signature FROM history_backfill WHERE user_id = ?", (user_id,)
            ).fetchone()
            if prior is not None and prior[0] == signature:
                return
            first_backfill = prior is None

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
            if "sessions" not in tables:
                return
            latest = history_conn.execute("SELECT MAX(updated_at) FROM sessions").fetchone()[0]
            cutoff = float(now if now is not None else time.time()) - RETENTION_DAYS * 86400
            turn_rows: list[sqlite3.Row] = []
            costs: dict[str, dict[str, Any]] = {}
            if "turns" in tables:
                turn_rows = history_conn.execute(
                    """
                    SELECT id, session_id, capability, status, created_at,
                           updated_at, finished_at
                    FROM turns
                    WHERE created_at >= ?
                    """,
                    (cutoff,),
                ).fetchall()
                # Cost extraction has to inspect JSON event metadata. Do that
                # once for historical data; later turns are captured directly
                # by the runtime hook, so admin refreshes stay proportional to
                # the small turns table rather than the full message history.
                if first_backfill and "messages" in tables:
                    costs = _cost_summaries_from_history(history_conn)
        finally:
            history_conn.close()

        timestamp = float(now if now is not None else time.time())
        with _connect() as activity_conn:
            if latest is not None:
                _upsert_timestamp(
                    activity_conn,
                    user_id=user_id,
                    username=username,
                    column="last_used_at",
                    timestamp=float(latest),
                )
            for row in turn_rows:
                usage = costs.get(str(row["id"]))
                activity_conn.execute(
                    """
                    INSERT INTO turn_activity (
                        turn_id, user_id, username, session_id, capability,
                        started_at, finished_at, status, llm_calls,
                        prompt_tokens, completion_tokens, total_tokens,
                        cost_usd, usage_reported
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        str(row["id"]),
                        user_id,
                        username,
                        str(row["session_id"] or ""),
                        str(row["capability"] or ""),
                        float(row["created_at"] or timestamp),
                        float(row["finished_at"] or row["updated_at"] or timestamp),
                        str(row["status"] or "completed"),
                        (usage or {}).get("total_calls", 0),
                        (usage or {}).get("prompt_tokens", 0),
                        (usage or {}).get("completion_tokens", 0),
                        (usage or {}).get("total_tokens", 0),
                        (usage or {}).get("total_cost_usd", 0.0),
                        1 if usage is not None else 0,
                    ),
                )
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
    except Exception:
        logger.warning("Could not backfill activity for %s", user_id, exc_info=True)


def _iso(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _usage_for(conn: sqlite3.Connection, user_id: str, cutoff: float) -> dict[str, Any]:
    turns = conn.execute(
        """
        SELECT
            COUNT(*) AS turns,
            COUNT(DISTINCT session_id) AS conversations,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN status IN ('failed', 'rejected') THEN 1 ELSE 0 END) AS failed,
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
        (user_id, _local_day(cutoff)),
    ).fetchone()[0]
    completed = _non_negative_int(turns["completed"])
    expected = _non_negative_int(turns["usage_expected"])
    reported = _non_negative_int(turns["usage_reported"])
    return {
        "conversations": _non_negative_int(turns["conversations"]),
        "turns": _non_negative_int(turns["turns"]),
        "completed_turns": completed,
        "failed_turns": _non_negative_int(turns["failed"]),
        "kb_queries": _non_negative_int(kb),
        "llm_calls": _non_negative_int(turns["llm_calls"]),
        "prompt_tokens": _non_negative_int(turns["prompt_tokens"]),
        "completion_tokens": _non_negative_int(turns["completion_tokens"]),
        "total_tokens": _non_negative_int(turns["total_tokens"]),
        "estimated_cost_usd": round(_non_negative_float(turns["cost_usd"]), 6),
        "usage_complete": expected == reported,
        "usage_reported_turns": reported,
    }


def get_activity_report(
    users: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Build the admin report and opportunistically backfill isolated history."""
    timestamp = float(now if now is not None else time.time())
    for user in users:
        backfill_user_history(user, now=timestamp)

    cutoff_7d = timestamp - 7 * 86400
    cutoff_30d = timestamp - 30 * 86400
    retention_cutoff = timestamp - RETENTION_DAYS * 86400
    local_now = datetime.fromtimestamp(timestamp).astimezone()
    start_today = local_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    results: list[dict[str, Any]] = []

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
                    "last_login_at": _iso(last_login),
                    "last_seen_at": _iso(last_seen),
                    "last_used_at": _iso(last_used),
                    "usage_7d": _usage_for(conn, user_id, cutoff_7d),
                    "usage_30d": _usage_for(conn, user_id, cutoff_30d),
                }
            )

    active_today = sum(
        1
        for user in results
        if user["last_used_at"]
        and datetime.fromisoformat(user["last_used_at"]).timestamp() >= start_today
    )
    active_7d = sum(
        1
        for user in results
        if user["last_used_at"]
        and datetime.fromisoformat(user["last_used_at"]).timestamp() >= cutoff_7d
    )
    inactive_30d = sum(
        1
        for user in results
        if not user["last_used_at"]
        or datetime.fromisoformat(user["last_used_at"]).timestamp() < cutoff_30d
    )
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
