"""Connection config for the Tencent IMA engine.

IMA credentials (``client_id`` + ``api_key``, issued at
https://ima.qq.com/agent-interface) identify an *account*, and a knowledge base
id identifies one of that account's libraries. The credentials therefore resolve
at two levels:

* **owner level** — ``settings/ima.json`` in the human account owner's
  workspace, edited under Knowledge → the IMA engine page. One pair is shared
  by that owner's ``ima`` KBs, but is never shared between users;
* **per KB** — the same two fields on the KB's ``kb_config.json`` entry. Present
  on knowledge bases connected before the engine page existed, and still the way
  to point one KB at a *different* IMA account.

The per-KB pair wins when it is complete, so an existing binding keeps working
untouched and rotating the account key updates every KB that relies on it.

This module is the single seam that reads that binding into a typed config; it
holds no global state and imports no HTTP client (the client lives in
``client.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from deeptutor.multi_user.paths import get_admin_path_service, get_owner_path_service
from deeptutor.services.config.runtime_settings import RuntimeSettingsService
from deeptutor.services.path_service import PathService

# IMA exposes exactly one retrieval call (``search_knowledge``) with no mode
# knob, so a KB bound to this engine has no per-KB search mode to pick. The
# empty tuple keeps the shared provider-mode plumbing happy while telling the
# UI there is nothing to offer.
SUPPORTED_MODES: tuple[str, ...] = ()
DEFAULT_MODE = ""


class ImaNotConfiguredError(RuntimeError):
    """Raised when a KB is missing the credentials or the knowledge base id."""


@dataclass(frozen=True)
class ImaCredentials:
    """One IMA account's credential pair."""

    client_id: str = ""
    api_key: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.client_id and self.api_key)


@dataclass(frozen=True)
class ImaConfig:
    """A KB's resolved connection to one Tencent IMA knowledge base."""

    client_id: str
    api_key: str
    knowledge_base_id: str


def get_ima_settings_service(*, kb_base_dir: str | Path | None = None) -> RuntimeSettingsService:
    """Return the settings store belonging to the relevant KB owner.

    With no explicit KB path this is the current human account owner and is the
    store used by the IMA settings page. During retrieval, callers pass the
    access-checked KB base directory so an assigned admin KB keeps using the
    administrator's account while a user's own KB uses that user's account.
    """
    if kb_base_dir is None:
        path_service = get_owner_path_service()
    else:
        # ``kb_base_dir`` is ``<workspace>/knowledge_bases``. Rebuilding a
        # PathService from its parent avoids depending on the requester's scope.
        workspace_root = Path(kb_base_dir).resolve().parent
        admin_service = get_admin_path_service()
        try:
            workspace_root.relative_to(admin_service.workspace_root.resolve() / "partners")
        except ValueError:
            path_service = PathService(workspace_root=workspace_root)
        else:
            # Partners are synthetic scopes owned by the administrator, not
            # credential-bearing accounts of their own.
            path_service = admin_service
    return RuntimeSettingsService.get_instance(path_service.get_settings_dir())


def _may_apply_process_overrides(service: RuntimeSettingsService) -> bool:
    """Deployment environment credentials belong only to the admin workspace."""
    try:
        admin_dir = get_admin_path_service().get_settings_dir().resolve()
        return service.settings_dir.resolve() == admin_dir
    except Exception:
        return False


def load_account_settings(
    *,
    kb_base_dir: str | Path | None = None,
    include_process_overrides: bool = True,
) -> dict[str, Any]:
    """Load one owner's IMA settings without leaking deployment env to users."""
    service = get_ima_settings_service(kb_base_dir=kb_base_dir)
    return service.load_ima(
        include_process_overrides=(
            include_process_overrides and _may_apply_process_overrides(service)
        )
    )


def save_account_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Persist IMA settings for the current human account owner."""
    return get_ima_settings_service().save_ima(settings)


def get_account_credentials(*, kb_base_dir: str | Path | None = None) -> ImaCredentials:
    """Load the KB owner's credential pair, or an empty one.

    Never raises: an unreadable settings file only means "not configured", which
    the callers already handle.
    """
    try:
        settings = load_account_settings(kb_base_dir=kb_base_dir)
    except Exception:
        return ImaCredentials()
    return ImaCredentials(
        client_id=str(settings.get("client_id") or "").strip(),
        api_key=str(settings.get("api_key") or "").strip(),
    )


def is_ima_configured() -> bool:
    """Whether account-level credentials are set (flags the engine as ready)."""
    return get_account_credentials().complete


def config_from_entry(
    entry: dict[str, Any],
    *,
    fallback: Optional[ImaCredentials] = None,
) -> ImaConfig:
    """Build an :class:`ImaConfig` from a ``kb_config.json`` KB entry.

    The entry's own credentials win; *fallback* (normally the account-level
    pair) fills in what it omits. Raises :class:`ImaNotConfiguredError` when any
    of the three required fields is still missing, so retrieval fails with a
    clear message instead of an opaque HTTP error from IMA.
    """
    client_id = str(entry.get("client_id") or "").strip()
    api_key = str(entry.get("api_key") or "").strip()
    knowledge_base_id = str(entry.get("knowledge_base_id") or "").strip()
    if fallback is not None:
        client_id = client_id or fallback.client_id
        api_key = api_key or fallback.api_key
    missing = [
        label
        for label, value in (
            ("client ID", client_id),
            ("API key", api_key),
            ("knowledge base ID", knowledge_base_id),
        )
        if not value
    ]
    if missing:
        raise ImaNotConfiguredError(
            "This knowledge base is not fully connected to Tencent IMA "
            f"(missing {', '.join(missing)}). Add the IMA credentials on the "
            "engine page under Knowledge, or re-create the knowledge base with "
            "complete credentials."
        )
    return ImaConfig(
        client_id=client_id,
        api_key=api_key,
        knowledge_base_id=knowledge_base_id,
    )


def resolve_kb_config(entry: dict[str, Any], *, kb_base_dir: str | Path | None = None) -> ImaConfig:
    """Resolve a KB entry with its owner's account credentials as fallback."""
    return config_from_entry(
        entry,
        fallback=get_account_credentials(kb_base_dir=kb_base_dir),
    )


__all__ = [
    "SUPPORTED_MODES",
    "DEFAULT_MODE",
    "ImaNotConfiguredError",
    "ImaCredentials",
    "ImaConfig",
    "config_from_entry",
    "get_account_credentials",
    "get_ima_settings_service",
    "is_ima_configured",
    "load_account_settings",
    "resolve_kb_config",
    "save_account_settings",
]
