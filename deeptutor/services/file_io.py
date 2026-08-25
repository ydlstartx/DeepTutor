"""Small, dependency-free helpers for durable service files."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any


def _atomic_replace(src: Path, dst: Path, *, max_retries: int = 5) -> None:
    """Replace *dst* with *src*, retrying on transient ``PermissionError``.

    On Windows the destination file is occasionally locked by another process
    (antivirus, indexer, or a concurrent reader), which makes ``Path.replace``
    raise ``PermissionError``. A short exponential backoff recovers in most
    cases without losing the already-written content.
    """
    delay = 0.2
    last_err: PermissionError | None = None
    for attempt in range(max_retries):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:  # pragma: no cover - platform specific
            last_err = exc
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 1.6)
    assert last_err is not None
    raise last_err


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as UTF-8 JSON without exposing a partial target file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        _atomic_replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with a same-directory atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        _atomic_replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def file_lock_exclusive(file_handle):
    """Acquire an exclusive (write) lock on a file - cross-platform."""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            file_handle.seek(0)
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


_lock_local = threading.local()


@contextmanager
def exclusive_write_lock(path: Path):
    """Serialize read-modify-write cycles on a JSON file across processes.

    Locks a sibling ``<name>.lock`` file whose inode is stable — locking the
    target file itself would not work with atomic replacement: a lock on the
    old inode excludes nobody opening the path afterwards.

    Reentrant within a thread (a higher-level transaction may hold the lock
    while a lower-level save acquires it again).
    """
    key = str(Path(path))
    held = getattr(_lock_local, "exclusive_write_locks", None)
    if held is None:
        held = _lock_local.exclusive_write_locks = set()
    if key in held:  # this thread already holds it
        yield
        return
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        with file_lock_exclusive(handle):
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)


__all__ = ["atomic_write_json", "atomic_write_text", "exclusive_write_lock", "file_lock_exclusive"]
