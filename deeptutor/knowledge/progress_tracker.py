"""
Progress Tracker - Tracks knowledge base initialization progress
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path

from deeptutor.logging import ALWAYS_CONSOLE_ATTR
from deeptutor.services.file_io import atomic_write_json

# Use unified logging system

_logger = logging.getLogger(__name__)


def _logger_instance():
    return _logger


def render_message_template(template: str, params: dict[str, object]) -> str:
    """Fill an i18next-style ``{{name}}`` template with *params*.

    Progress lines are shown verbatim in the web log box, so they have to be
    translatable — but the backend has no viewer language (indexing runs as a
    detached task, and the language of record lives in the browser). So the
    wire carries the English template plus its values and the frontend renders
    it with ``t()``; this produces the English fallback for every other
    consumer, from the same single string.
    """
    text = template
    for name, value in params.items():
        text = text.replace("{{" + name + "}}", str(value))
    return text


class ProgressStage(Enum):
    """Initialization stage"""

    INITIALIZING = "initializing"  # Initializing
    PROCESSING_DOCUMENTS = "processing_documents"  # Processing documents
    PROCESSING_FILE = "processing_file"  # Processing single file
    COMPLETED = "completed"  # Completed
    ERROR = "error"  # Error


class ProgressTracker:
    """Progress tracker"""

    def __init__(self, kb_name: str, base_dir: Path):
        self.kb_name = kb_name
        self.base_dir = base_dir
        self.kb_dir = base_dir / kb_name
        self.progress_file = self.kb_dir / ".progress.json"
        self._callbacks: list = []  # Support multiple callbacks
        self.task_id: str | None = None  # Task ID (for log identification)
        self._console_progress: dict[tuple[str, str], int] = {}

    def _is_console_milestone(
        self,
        *,
        stage: ProgressStage,
        message: str,
        message_key: str | None,
        current: int,
        total: int,
        error: str | None,
    ) -> bool:
        """Keep long-running KB tasks observable without flooding stdout."""
        if error or stage in {ProgressStage.COMPLETED, ProgressStage.ERROR}:
            return True
        if total <= 0:
            return True

        percent = int(current / total * 100)
        phase = (stage.value, message_key or message)
        previous = self._console_progress.get(phase)
        visible = previous is None or current in {0, 1, total} or percent >= previous + 5
        if visible:
            self._console_progress[phase] = percent
        return visible

    def set_callback(self, callback: Callable[[dict], None]):
        """Set progress callback function (can be called multiple times to add multiple callbacks)"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[dict], None]):
        """Remove progress callback function"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify(self, progress: dict):
        """Notify progress update (call all callbacks)"""
        from deeptutor.runtime.mode import is_server

        if is_server():
            try:
                from deeptutor.api.utils.progress_broadcaster import ProgressBroadcaster

                broadcaster = ProgressBroadcaster.get_instance()

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(broadcaster.broadcast(self.kb_name, progress))
                except RuntimeError:
                    pass
            except (ImportError, Exception):
                pass

        for callback in self._callbacks:
            try:
                callback(progress)
            except Exception as e:
                _logger_instance().debug("Progress callback error: %s", e)

    def _save_progress(self, progress: dict) -> bool:
        """Save progress to kb_config.json and local .progress.json file.

        Returns False when the central write was rejected because another
        task now owns the KB — the caller must then skip the snapshot and
        the broadcast, or the UI would still show the superseded task's
        progress from ``.progress.json``.
        """
        from deeptutor.knowledge.policy import is_kb_query_only

        if is_kb_query_only():
            _logger_instance().warning(
                "Dropped progress write for '%s': knowledge bases are query-only",
                self.kb_name,
            )
            return False

        accepted = True
        # Save to kb_config.json (centralized config)
        try:
            from deeptutor.knowledge.manager import KnowledgeBaseManager

            manager = KnowledgeBaseManager(base_dir=str(self.base_dir))

            # Determine status based on stage
            stage = progress.get("stage", "")
            if stage == "completed":
                status = "ready"
            elif stage == "error":
                status = "error"
            elif stage in [
                "initializing",
                "processing_documents",
                "processing_file",
            ]:
                status = "processing"
            else:
                status = "initializing"

            # A task only publishes while it still owns the KB entry: when a
            # newer task (upload/sync/rebuild) has claimed it, this task's
            # late progress/completion must not clobber the new state. The
            # claim itself happens at task-start call sites (router /
            # initializer), not here.
            task_id = progress.get("task_id")
            only_if_task = None
            if task_id:
                from deeptutor.knowledge.manager import get_process_identity

                pid, instance_id, _ = get_process_identity()
                only_if_task = (pid, instance_id, task_id)

            # Update kb_config.json with status and progress
            written = manager.update_kb_status(
                name=self.kb_name,
                status=status,
                progress={
                    "stage": progress.get("stage"),
                    "message": progress.get("message"),
                    "percent": progress.get("progress_percent", 0),
                    "current": progress.get("current", 0),
                    "total": progress.get("total", 0),
                    "file_name": progress.get("file_name"),
                    "error": progress.get("error"),
                    "error_code": progress.get("error_code"),
                    "retryable": progress.get("retryable"),
                    "timestamp": progress.get("timestamp"),
                    "task_id": progress.get("task_id"),
                    "indexed_count": progress.get("indexed_count"),
                    "index_changed": progress.get("index_changed"),
                    "index_action": progress.get("index_action"),
                    # Carried alongside the rendered English so a page reload
                    # can still translate the last line it shows.
                    "message_key": progress.get("message_key"),
                    "message_params": progress.get("message_params"),
                },
                only_if_task=only_if_task,
            )
            if not written:
                accepted = False
                _logger_instance().warning(
                    "Progress update from task %s dropped: KB '%s' is now owned by another task",
                    task_id,
                    self.kb_name,
                )
        except Exception as e:
            _logger_instance().warning("Failed to save progress to kb_config.json: %s", e)

        if accepted:
            # Persist the last seen progress snapshot so websocket subscribers and
            # page reloads can recover the live state without relying on in-memory callbacks.
            try:
                atomic_write_json(self.progress_file, progress)
            except Exception as e:
                _logger_instance().warning(
                    "Failed to persist progress snapshot for '%s': %s", self.kb_name, e
                )
        return accepted

    def update(
        self,
        stage: ProgressStage,
        message: str = "",
        current: int = 0,
        total: int = 0,
        file_name: str = "",
        error: str | None = None,
        error_code: str | None = None,
        retryable: bool | None = None,
        indexed_count: int | None = None,
        index_changed: bool | None = None,
        index_action: str | None = None,
        message_key: str | None = None,
        message_params: dict[str, object] | None = None,
    ):
        """Update progress.

        Pass ``message_key`` (an English ``{{name}}`` template) plus
        ``message_params`` instead of a pre-formatted ``message`` so the web log
        box can translate the line; ``message`` is then rendered from the same
        template for every consumer that has no i18n of its own.
        """
        params = message_params or {}
        if message_key and not message:
            message = render_message_template(message_key, params)
        progress: dict[str, object] = {
            "kb_name": self.kb_name,
            "task_id": self.task_id,
            "stage": stage.value,
            "message": message,
            "current": current,
            "total": total,
            "file_name": file_name,
            "progress_percent": int(current / total * 100) if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }
        if indexed_count is not None:
            progress["indexed_count"] = indexed_count
        if index_changed is not None:
            progress["index_changed"] = index_changed
        if index_action:
            progress["index_action"] = index_action
        if message_key:
            progress["message_key"] = message_key
            progress["message_params"] = params

        if error:
            progress["error"] = error
            progress["stage"] = ProgressStage.ERROR.value
        if error_code:
            progress["error_code"] = error_code
        if retryable is not None:
            progress["retryable"] = retryable

        # Output to logger (terminal and log file)
        try:
            logger = _logger_instance()
            prefix = f"[{self.task_id}]" if self.task_id else ""
            always_console = self._is_console_milestone(
                stage=stage,
                message=message,
                message_key=message_key,
                current=current,
                total=total,
                error=error,
            )
            log_extra = {ALWAYS_CONSOLE_ATTR: always_console}

            if total > 0:
                percent = progress["progress_percent"]
                progress_msg = f"{prefix} {message} ({current}/{total}, {percent}%)"
                if file_name:
                    progress_msg += f" - File: {file_name}"
            else:
                progress_msg = f"{prefix} {message}"
                if file_name:
                    progress_msg += f" - File: {file_name}"

            if error:
                logger.error(f"{progress_msg} - Error: {error}", extra=log_extra)
            else:
                logger.info(progress_msg, extra=log_extra)
        except Exception:
            # If unified logging fails unexpectedly, use stdlib logger as fallback.
            fallback_logger = logging.getLogger("deeptutor.ProgressTracker")
            prefix = f"[{self.task_id}]" if self.task_id else ""
            fallback_logger.warning(
                "%s [ProgressTracker] %s (%s/%s)",
                prefix,
                message,
                current,
                total if total > 0 else "?",
            )
            if error:
                fallback_logger.error("%s [ProgressTracker] Error: %s", prefix, error)

        if not self._save_progress(progress):
            # Superseded by a newer task — don't emit or broadcast its state.
            return

        if self.task_id:
            try:
                from deeptutor.api.utils.task_log_stream import get_task_stream_manager

                get_task_stream_manager().emit(self.task_id, "progress", progress)
            except Exception as e:
                _logger_instance().debug("Failed to emit task progress event: %s", e)

        self._notify(progress)

    def get_progress(self) -> dict | None:
        """Get current progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _logger_instance().debug(f"Failed to read progress file for '{self.kb_name}': {e}")

        try:
            from deeptutor.knowledge.manager import KnowledgeBaseManager

            manager = KnowledgeBaseManager(base_dir=str(self.base_dir))
            status = manager.get_kb_status(self.kb_name)
            if status and status.get("progress"):
                return status.get("progress")
        except Exception as e:
            _logger_instance().debug(
                "Failed to recover progress snapshot from kb_config for '%s': %s",
                self.kb_name,
                e,
            )

        return None

    def clear(self):
        """Clear progress file"""
        from deeptutor.knowledge.policy import ensure_kb_write_allowed

        ensure_kb_write_allowed()
        if self.progress_file.exists():
            try:
                self.progress_file.unlink()
            except Exception as e:
                _logger_instance().debug(f"Failed to clear progress file for '{self.kb_name}': {e}")
