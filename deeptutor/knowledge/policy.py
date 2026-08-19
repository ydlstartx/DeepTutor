"""Deployment policy for build-disabled, query-only knowledge-base servers."""

from __future__ import annotations

KB_QUERY_ONLY_ENV = "DEEPTUTOR_KB_QUERY_ONLY"
KB_QUERY_ONLY_MESSAGE = (
    "Knowledge base modification is disabled on this server. "
    "This deployment is configured for query-only access."
)


class KnowledgeBaseWriteDisabledError(PermissionError):
    """Raised when a KB mutation is attempted on a query-only deployment."""


def is_kb_query_only() -> bool:
    """Return the effective deployment policy from centralized runtime settings."""
    from deeptutor.services.config.runtime_settings import load_system_settings

    return bool(load_system_settings().get("kb_query_only", False))


def ensure_kb_write_allowed() -> None:
    """Reject a knowledge-base mutation when the server is query-only."""
    if is_kb_query_only():
        raise KnowledgeBaseWriteDisabledError(KB_QUERY_ONLY_MESSAGE)


def ensure_kb_delete_allowed(*, is_admin: bool) -> None:
    """Allow query-only deletion only for the current administrator.

    Deletion is an operational cleanup action rather than knowledge-base
    construction.  Administrators may remove published local KBs or connected
    pointers; ordinary users remain subject to the query-only write guard.
    """
    if is_kb_query_only() and not is_admin:
        raise KnowledgeBaseWriteDisabledError(KB_QUERY_ONLY_MESSAGE)


__all__ = [
    "KB_QUERY_ONLY_ENV",
    "KB_QUERY_ONLY_MESSAGE",
    "KnowledgeBaseWriteDisabledError",
    "ensure_kb_delete_allowed",
    "ensure_kb_write_allowed",
    "is_kb_query_only",
]
