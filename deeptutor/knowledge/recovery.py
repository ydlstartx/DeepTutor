"""Startup recovery for interrupted knowledge-base tasks.

Indexing tasks (``kb_init_*`` / ``kb_upload_*`` / ``kb_reindex_*``) served by
the API run inside the API process. When the process stops mid-task —
Ctrl-C, kill, OOM — the task dies without promoting the KB out of its live
status, leaving a zombie: the UI shows "processing" forever even though
nothing is running (observed after a stalled proxy connection hung an
indexing task for 80+ minutes).

A live status alone does NOT prove a zombie — a build may legitimately be
running in another process (``deeptutor kb create`` CLI, a sibling uvicorn
worker, the old process of a rolling deploy). Three mechanisms keep
recovery from touching live or already-finished work:

* Every live-status write stamps the entry with the owner's pid, process
  create time, and a per-boot instance id (see
  ``KnowledgeBaseManager.update_kb_status``). Recovery skips candidates
  whose owner is still alive — verified via psutil (``os.kill(pid, 0)`` is
  unsafe on Windows, where it would terminate the process). PID reuse is
  caught by comparing the stamped create time. Entries without a stamp
  predate the mechanism and are treated as zombies.
* Writes use ``only_if_status_in`` + ``only_if_task`` — a compare-and-set
  under the manager's cross-process write lock that also matches the task
  identity (pid + boot instance + task id) seen at scan time. A task that
  finished, or a NEW task started after the scan — even in the same
  process, same live status — is never overwritten.
* Recovery always resolves to ``error``, never to ``ready``: an on-disk
  ready index cannot prove the interrupted task finished (an old index
  says nothing about documents the task was still adding), and silently
  showing "ready" for half-indexed content is worse than asking for a
  re-index. The read path's #418 reconciliation keeps working for the
  not-yet-restarted window.

Cross-container note: with a shared data volume, a rolling deploy's old
container is invisible to psutil in the new one, so its in-flight task
looks dead and gets reset to ``error``; when the old task finishes it
promotes the KB to ``ready``, so the state self-heals. A shared
heartbeat/lease would be needed to do better. Likewise, "owner process
alive" cannot prove the task inside it is still running — a task that
dies without writing its status (while its worker lives on) stays
skipped until that process exits. Task heartbeats would close that gap;
progress timestamps are too sparse to serve as one.

Multi-user deployments keep per-user KB trees under ``data/users/<uid>``;
:func:`recover_all_interrupted_kb_tasks` covers the admin tree plus every
user tree. Partner workspaces never run indexing and are not scanned.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from deeptutor.knowledge.manager import LIVE_STATUSES, owner_process_alive

logger = logging.getLogger(__name__)


def recover_interrupted_kb_tasks(base_dir: str | Path) -> list[str]:
    """Reset zombie "live" KBs under ``base_dir`` to ``error``.

    Indexed data is never touched; re-indexing resumes from the raw files.
    Returns the recovered KB names.
    """
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = Path(base_dir)
    if not base.is_dir():
        return []
    manager = KnowledgeBaseManager(base_dir=str(base))
    entries = dict(manager.config.get("knowledge_bases") or {})

    recovered: list[str] = []
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "")
        if status not in LIVE_STATUSES:
            continue
        if owner_process_alive(entry) is True:
            # Genuinely running in another process (CLI, sibling worker,
            # rolling deploy's old process) — not a zombie.
            continue

        # Connected KBs store their data under an external path; resolve the
        # KB directory for the progress file cleanup.
        kb_rel = Path(str(entry.get("path") or name))
        kb_dir = kb_rel if kb_rel.is_absolute() else base / kb_rel

        written = manager.update_kb_status(
            name=name,
            status="error",
            progress={
                "stage": "error",
                "message": "Indexing was interrupted by a backend stop/restart",
                "percent": 0,
                "current": 0,
                "total": 1,
                "file_name": "",
                "error": (
                    "The backend stopped or the task stalled while indexing. "
                    "Indexed data was left untouched; re-index to continue."
                ),
                "timestamp": datetime.now().isoformat(),
            },
            # Compare-and-set on status AND task identity: a task that
            # finished — or a new one started after the scan, even in the
            # same process (same live status, same owner, new task id) —
            # must not be overwritten.
            only_if_status_in=tuple(sorted(LIVE_STATUSES)),
            only_if_task=(
                entry.get("task_owner_pid"),
                entry.get("task_owner_instance"),
                entry.get("task_owner_task_id"),
            ),
        )
        if not written:
            logger.info("KB '%s' changed state during recovery; skipping", name)
            continue

        progress_file = kb_dir / ".progress.json"
        try:
            progress_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove stale progress file %s: %s", progress_file, exc)

        recovered.append(name)
        logger.warning(
            "KB '%s' was stuck in '%s' from a dead run; reset to error for re-indexing",
            name,
            status,
        )
    return recovered


def recover_all_interrupted_kb_tasks(data_root: str | Path) -> list[str]:
    """Recover zombie KBs in the admin tree and every per-user tree.

    ``data_root`` is the runtime ``data/`` directory: the admin KB tree is
    ``<data>/knowledge_bases`` and each non-admin user's tree is
    ``<data>/users/<uid>/knowledge_bases``. Returns display labels —
    ``name`` for admin KBs, ``<uid>:name`` for user KBs.
    """
    root = Path(data_root)
    roots: list[tuple[str, Path]] = [("", root / "knowledge_bases")]
    users_root = root / "users"
    if users_root.is_dir():
        for workspace in sorted(users_root.iterdir()):
            if workspace.is_dir():
                roots.append((f"{workspace.name}:", workspace / "knowledge_bases"))

    recovered: list[str] = []
    for prefix, base in roots:
        for name in recover_interrupted_kb_tasks(base):
            recovered.append(f"{prefix}{name}")
    return recovered


__all__ = [
    "LIVE_STATUSES",
    "recover_all_interrupted_kb_tasks",
    "recover_interrupted_kb_tasks",
]
