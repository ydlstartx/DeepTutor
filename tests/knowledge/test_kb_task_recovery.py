"""Startup recovery for interrupted KB tasks (``knowledge/recovery.py``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from deeptutor.knowledge.recovery import (
    recover_all_interrupted_kb_tasks,
    recover_interrupted_kb_tasks,
)


def _write_config(base: Path, entries: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "kb_config.json").write_text(json.dumps({"knowledge_bases": entries}))


def _read_entry(base: Path, name: str) -> dict:
    cfg = json.loads((base / "kb_config.json").read_text())
    return cfg["knowledge_bases"][name]


def test_zombie_processing_kb_is_reset_to_error(tmp_path) -> None:
    base = tmp_path / "kbs"
    kb = base / "alpha"
    kb.mkdir(parents=True)
    (kb / ".progress.json").write_text("{}")
    _write_config(
        base,
        {
            "alpha": {
                "path": "alpha",
                "status": "processing",
                "progress": {"stage": "processing_documents"},
            }
        },
    )

    recovered = recover_interrupted_kb_tasks(base)

    assert recovered == ["alpha"]
    entry = _read_entry(base, "alpha")
    assert entry["status"] == "error"
    assert entry["progress"]["stage"] == "error"
    assert entry["last_error"]
    assert not (kb / ".progress.json").exists()


def test_initializing_kb_is_also_recovered(tmp_path) -> None:
    base = tmp_path / "kbs"
    (base / "beta").mkdir(parents=True)
    _write_config(base, {"beta": {"path": "beta", "status": "initializing"}})

    assert recover_interrupted_kb_tasks(base) == ["beta"]
    assert _read_entry(base, "beta")["status"] == "error"


def test_ready_kb_is_untouched(tmp_path) -> None:
    base = tmp_path / "kbs"
    (base / "gamma").mkdir(parents=True)
    _write_config(base, {"gamma": {"path": "gamma", "status": "ready"}})

    assert recover_interrupted_kb_tasks(base) == []
    assert _read_entry(base, "gamma")["status"] == "ready"


def test_connected_kb_progress_file_outside_base_dir(tmp_path) -> None:
    external = tmp_path / "external-kb"
    external.mkdir()
    (external / ".progress.json").write_text("{}")
    base = tmp_path / "kbs"
    _write_config(base, {"ext": {"path": str(external), "status": "processing"}})

    assert recover_interrupted_kb_tasks(base) == ["ext"]
    assert not (external / ".progress.json").exists()
    assert _read_entry(base, "ext")["status"] == "error"


def test_missing_base_dir_returns_empty(tmp_path) -> None:
    assert recover_interrupted_kb_tasks(tmp_path / "nope") == []
    assert not (tmp_path / "nope").exists()  # must not create anything


def test_status_flip_between_scan_and_write_is_not_clobbered(tmp_path, monkeypatch) -> None:
    """A build that finishes (ready) between the recovery scan and the
    recovery write must survive — the update is compare-and-set guarded."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "processing"}})

    original = KnowledgeBaseManager._load_config
    calls = 0

    def flip_after_scan(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(self)  # the scan still sees "processing"
        # ...but by write time the other task has already finished:
        return {"knowledge_bases": {"alpha": {"path": "alpha", "status": "ready"}}}

    monkeypatch.setattr(KnowledgeBaseManager, "_load_config", flip_after_scan)

    assert recover_interrupted_kb_tasks(base) == []
    # The manager backfills/normalizes entries on any load (pre-existing
    # behavior), so compare state, not bytes: status must still be the
    # pre-recovery one with no error fields written.
    entry = _read_entry(base, "alpha")
    assert entry["status"] == "processing"
    assert "last_error" not in entry
    assert (entry.get("progress") or {}).get("stage") != "error"


def test_update_kb_status_only_if_guard(tmp_path) -> None:
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))

    assert manager.update_kb_status("alpha", "error", only_if_status_in=("processing",)) is False
    assert _read_entry(base, "alpha")["status"] == "ready"
    assert manager.update_kb_status("alpha", "error", only_if_status_in=("ready",)) is True
    assert _read_entry(base, "alpha")["status"] == "error"


def test_recover_all_covers_admin_and_user_trees(tmp_path) -> None:
    data = tmp_path / "data"
    _write_config(
        data / "knowledge_bases", {"admin_kb": {"path": "admin_kb", "status": "processing"}}
    )
    _write_config(
        data / "users" / "u1" / "knowledge_bases", {"kb1": {"path": "kb1", "status": "processing"}}
    )
    _write_config(
        data / "users" / "u2" / "knowledge_bases", {"kb2": {"path": "kb2", "status": "ready"}}
    )

    recovered = recover_all_interrupted_kb_tasks(data)

    assert sorted(recovered) == ["admin_kb", "u1:kb1"]
    assert _read_entry(data / "knowledge_bases", "admin_kb")["status"] == "error"
    assert _read_entry(data / "users" / "u1" / "knowledge_bases", "kb1")["status"] == "error"
    assert _read_entry(data / "users" / "u2" / "knowledge_bases", "kb2")["status"] == "ready"


def test_recover_all_without_users_root(tmp_path) -> None:
    data = tmp_path / "data"
    _write_config(data / "knowledge_bases", {"solo": {"path": "solo", "status": "processing"}})

    assert recover_all_interrupted_kb_tasks(data) == ["solo"]


# --------------------------------------------------------------------------- #
# task ownership (task_owner_pid)
# --------------------------------------------------------------------------- #


def test_live_owner_process_is_not_recovered(tmp_path) -> None:
    """A live status whose owner process is still running (CLI build, sibling
    uvicorn worker, old process of a rolling deploy) is not a zombie."""
    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(
        base,
        {"alpha": {"path": "alpha", "status": "processing", "task_owner_pid": os.getpid()}},
    )

    assert recover_interrupted_kb_tasks(base) == []
    assert _read_entry(base, "alpha")["status"] == "processing"


def test_dead_owner_process_is_recovered(tmp_path) -> None:
    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()  # reaped: pid is dead
    _write_config(
        base,
        {"alpha": {"path": "alpha", "status": "processing", "task_owner_pid": child.pid}},
    )

    assert recover_interrupted_kb_tasks(base) == ["alpha"]
    assert _read_entry(base, "alpha")["status"] == "error"


def test_live_status_stamps_and_clears_owner_identity(tmp_path) -> None:
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))

    manager.update_kb_status(
        "alpha", "processing", progress={"stage": "processing_documents", "task_id": "kb_upload_1"}
    )
    entry = _read_entry(base, "alpha")
    assert entry["task_owner_pid"] == os.getpid()
    assert entry["task_owner_instance"]  # per-boot uuid
    assert entry["task_owner_started"] > 0  # process create time
    assert entry["task_owner_task_id"] == "kb_upload_1"
    manager.update_kb_status("alpha", "ready")
    entry = _read_entry(base, "alpha")
    for key in (
        "task_owner_pid",
        "task_owner_instance",
        "task_owner_started",
        "task_owner_task_id",
    ):
        assert key not in entry


def test_reused_pid_is_detected_via_create_time(tmp_path) -> None:
    """Same pid but a different create time = the old owner died and the pid
    was recycled (common with small container pids) — treat as zombie."""
    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(
        base,
        {
            "alpha": {
                "path": "alpha",
                "status": "processing",
                "task_owner_pid": os.getpid(),  # alive...
                "task_owner_started": 0.0,  # ...but stamped by somebody else
            }
        },
    )

    assert recover_interrupted_kb_tasks(base) == ["alpha"]
    assert _read_entry(base, "alpha")["status"] == "error"


def test_new_task_with_same_status_is_not_clobbered(tmp_path, monkeypatch) -> None:
    """Scan sees dead owner A; before the write, another process starts a new
    task (same live status, different owner B). The owner-aware CAS must skip."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_config(
        base,
        {"alpha": {"path": "alpha", "status": "processing", "task_owner_pid": dead.pid}},
    )

    original = KnowledgeBaseManager._load_config
    calls = 0

    def new_task_started_after_scan(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(self)  # scan: dead owner A
        return {
            "knowledge_bases": {
                "alpha": {
                    "path": "alpha",
                    "status": "processing",  # still live, but a NEW task
                    "task_owner_pid": os.getpid(),
                    "task_owner_instance": "other-boot",
                }
            }
        }

    monkeypatch.setattr(KnowledgeBaseManager, "_load_config", new_task_started_after_scan)

    assert recover_interrupted_kb_tasks(base) == []
    entry = _read_entry(base, "alpha")
    assert entry["status"] == "processing"
    assert "last_error" not in entry


def test_interrupted_task_errors_even_with_old_index(tmp_path) -> None:
    """A ready index from an earlier build must NOT shield an interrupted
    task: documents it was adding may not be searchable, so the honest
    outcome is error (prompting re-index), never a silent ready."""
    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "processing"}})

    assert recover_interrupted_kb_tasks(base) == ["alpha"]
    assert _read_entry(base, "alpha")["status"] == "error"


def test_new_task_id_same_owner_is_not_clobbered(tmp_path, monkeypatch) -> None:
    """The task id in the CAS is the discriminator when owner pid/instance
    match the scan: a different persisted task id means the live status now
    belongs to a NEWER task, which recovery must not overwrite."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_config(
        base,
        {
            "alpha": {
                "path": "alpha",
                "status": "processing",
                "task_owner_pid": dead.pid,
                "task_owner_instance": "boot-a",
                "task_owner_task_id": "kb_upload_old",
            }
        },
    )

    original = KnowledgeBaseManager._load_config
    calls = 0

    def new_task_after_scan(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(self)  # scan: the old (dead-owner) task
        return {
            "knowledge_bases": {
                "alpha": {
                    "path": "alpha",
                    "status": "processing",
                    "task_owner_pid": dead.pid,
                    "task_owner_instance": "boot-a",
                    "task_owner_task_id": "kb_upload_new",  # same owner, NEW task
                }
            }
        }

    monkeypatch.setattr(KnowledgeBaseManager, "_load_config", new_task_after_scan)

    assert recover_interrupted_kb_tasks(base) == []
    entry = _read_entry(base, "alpha")
    assert entry["status"] == "processing"
    assert entry["task_owner_task_id"] == "kb_upload_old"
    assert "last_error" not in entry


def test_concurrent_status_updates_serialize_without_corruption(tmp_path) -> None:
    """Two writers (e.g. API + CLI) racing on one config file: the write lock
    keeps every update atomic and the file always parses."""
    import threading

    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "processing"}})
    managers = [KnowledgeBaseManager(base_dir=str(base)) for _ in range(2)]

    def writer(manager, status) -> None:
        for _ in range(10):
            manager.update_kb_status("alpha", status)

    threads = [
        threading.Thread(target=writer, args=(managers[0], "processing")),
        threading.Thread(target=writer, args=(managers[1], "error")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _read_entry(base, "alpha")["status"] in {"processing", "error"}


def test_mutators_do_not_clobber_concurrent_status_updates(tmp_path) -> None:
    """A mutator holding a stale snapshot must not overwrite a fresh status
    on save — every mutator reloads under the write lock (transact)."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    (base / "beta").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    writer = KnowledgeBaseManager(base_dir=str(base))
    stale = KnowledgeBaseManager(base_dir=str(base))
    stale.config = stale._load_config()  # snapshot taken before the update

    writer.update_kb_status("alpha", "processing", progress={"task_id": "t-1"})
    stale.register_knowledge_base("beta")  # must not resurrect alpha=ready

    config = json.loads((base / "kb_config.json").read_text())["knowledge_bases"]
    assert config["alpha"]["status"] == "processing"
    assert "beta" in config


def test_superseded_task_progress_is_dropped(tmp_path) -> None:
    """Task A owns the KB; task B claims it; A's late progress and terminal
    writes must not clobber B's state."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager
    from deeptutor.knowledge.progress_tracker import ProgressStage, ProgressTracker

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))

    # Both tasks claim at start (unconditional task-start writes).
    manager.update_kb_status("alpha", "processing", progress={"task_id": "task-a"})
    manager.update_kb_status("alpha", "processing", progress={"task_id": "task-b"})

    # A's late progress via its tracker is owned-by-B now → dropped.
    tracker_a = ProgressTracker("alpha", base)
    tracker_a.task_id = "task-a"
    tracker_a.update(ProgressStage.PROCESSING_DOCUMENTS, "stale work", current=1, total=3)

    entry = _read_entry(base, "alpha")
    assert entry["status"] == "processing"
    assert entry["progress"].get("task_id") == "task-b"

    # B's own tracker still publishes normally.
    tracker_b = ProgressTracker("alpha", base)
    tracker_b.task_id = "task-b"
    tracker_b.update(ProgressStage.PROCESSING_DOCUMENTS, "live work", current=2, total=3)
    assert _read_entry(base, "alpha")["progress"].get("message") == "live work"


# --------------------------------------------------------------------------- #
# task claiming + terminal-state ownership
# --------------------------------------------------------------------------- #


def test_claim_rejected_while_live_owner_runs(tmp_path) -> None:
    """A live status whose owner process is alive cannot be claimed — this is
    what makes the API answer 409 instead of running two indexing tasks."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))

    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "t-a"}) is True
    # Owner (this process) is alive → a second claim is refused.
    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "t-b"}) is False
    assert _read_entry(base, "alpha")["task_owner_task_id"] == "t-a"


def test_claim_takeover_from_dead_owner(tmp_path) -> None:
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_config(
        base,
        {"alpha": {"path": "alpha", "status": "processing", "task_owner_pid": dead.pid}},
    )
    manager = KnowledgeBaseManager(base_dir=str(base))

    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "t-new"}) is True
    assert _read_entry(base, "alpha")["task_owner_task_id"] == "t-new"


def test_late_terminal_write_from_superseded_task_is_rejected(tmp_path) -> None:
    """B took over from dead A and completed; A's late terminal write must
    not flip B's ready back to error nor overwrite completion metadata."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager, get_process_identity

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_config(
        base,
        {
            "alpha": {
                "path": "alpha",
                "status": "processing",
                "task_owner_pid": dead.pid,
                "task_owner_instance": "boot-old",
                "task_owner_task_id": "task-a",
            }
        },
    )
    manager = KnowledgeBaseManager(base_dir=str(base))
    pid, instance, _ = get_process_identity()

    # B takes over from the dead owner and completes.
    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "task-b"})
    assert manager.update_kb_status(
        "alpha",
        "ready",
        progress={"task_id": "task-b", "indexed_count": 7, "index_changed": True},
        only_if_task=(pid, instance, "task-b"),
    )

    # A's zombie write arrives after B completed → rejected.
    assert (
        manager.update_kb_status(
            "alpha",
            "error",
            progress={"task_id": "task-a", "error": "late failure"},
            only_if_task=(dead.pid, "boot-old", "task-a"),
        )
        is False
    )
    entry = _read_entry(base, "alpha")
    assert entry["status"] == "ready"
    assert entry["last_indexed_count"] == 7
    assert entry["last_task_id"] == "task-b"


def test_same_task_double_terminal_write_still_passes(tmp_path) -> None:
    """The tracker writes completed, then the router writes the final ready —
    one task, two terminal writes; the second must pass."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager, get_process_identity

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))
    pid, instance, _ = get_process_identity()
    owner = (pid, instance, "task-a")

    manager.claim_kb_task("alpha", "processing", progress={"task_id": "task-a"})
    assert manager.update_kb_status(
        "alpha", "ready", progress={"task_id": "task-a"}, only_if_task=owner
    )
    # Owner is cleared now, but last_task_id == task-a → follow-up passes.
    assert manager.update_kb_status(
        "alpha",
        "ready",
        progress={"task_id": "task-a", "indexed_count": 3, "index_changed": True},
        only_if_task=owner,
    )
    assert _read_entry(base, "alpha")["last_indexed_count"] == 3


def test_rejected_progress_skips_snapshot_and_broadcast(tmp_path) -> None:
    """A superseded task's update must not touch .progress.json nor fire
    callbacks — otherwise the UI still shows the stale task's progress."""
    from deeptutor.knowledge.manager import KnowledgeBaseManager
    from deeptutor.knowledge.progress_tracker import ProgressStage, ProgressTracker

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))
    manager.claim_kb_task("alpha", "processing", progress={"task_id": "task-b"})

    tracker_a = ProgressTracker("alpha", base)
    tracker_a.task_id = "task-a"
    seen: list[dict] = []
    tracker_a.set_callback(seen.append)
    tracker_a.update(ProgressStage.PROCESSING_DOCUMENTS, "stale", current=1, total=2)

    assert seen == []  # no broadcast
    assert not (base / "alpha" / ".progress.json").exists()  # no snapshot


def test_exclusive_write_lock_creates_missing_parent_dirs(tmp_path) -> None:
    from deeptutor.services.file_io import exclusive_write_lock

    target = tmp_path / "brand" / "new" / "kb_config.json"
    with exclusive_write_lock(target):
        pass
    assert (tmp_path / "brand" / "new" / "kb_config.json.lock").exists()


def test_cli_add_refuses_busy_kb(tmp_path) -> None:
    """CLI `kb add` shares the API's atomic claim: a live-owned KB fails
    loudly instead of running two indexing tasks over one index."""
    import asyncio

    from deeptutor.knowledge.add_documents import add_documents
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))
    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "t-1"})

    with pytest.raises(RuntimeError, match="already being processed"):
        asyncio.run(add_documents("alpha", ["x.pdf"], base_dir=str(base)))


def test_cli_initialize_refuses_busy_kb(tmp_path) -> None:
    import asyncio

    from deeptutor.knowledge.initializer import initialize_knowledge_base
    from deeptutor.knowledge.manager import KnowledgeBaseManager

    base = tmp_path / "kbs"
    (base / "alpha").mkdir(parents=True)
    _write_config(base, {"alpha": {"path": "alpha", "status": "ready"}})
    manager = KnowledgeBaseManager(base_dir=str(base))
    assert manager.claim_kb_task("alpha", "processing", progress={"task_id": "t-1"})

    with pytest.raises(RuntimeError, match="already being processed"):
        asyncio.run(initialize_knowledge_base("alpha", ["x.pdf"], base_dir=str(base)))
