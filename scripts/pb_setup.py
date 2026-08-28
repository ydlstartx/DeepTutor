#!/usr/bin/env python3
"""
PocketBase collection bootstrap script.

Run this once after starting PocketBase for the first time:

    python scripts/pb_setup.py

Requires integrations.pocketbase_url, integrations.pocketbase_admin_email, and
integrations.pocketbase_admin_password in data/user/settings/integrations.json.

Safe to re-run — existing records are preserved and missing fields are added.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deeptutor.services.config import load_integrations_settings

_INTEGRATIONS = load_integrations_settings()
POCKETBASE_BASE_URL = str(_INTEGRATIONS["pocketbase_url"]).rstrip("/")
ADMIN_EMAIL = str(_INTEGRATIONS["pocketbase_admin_email"])
ADMIN_PASSWORD = str(_INTEGRATIONS["pocketbase_admin_password"])


def _require_env():
    missing = []
    if not POCKETBASE_BASE_URL:
        missing.append("integrations.pocketbase_url")
    if not ADMIN_EMAIL:
        missing.append("integrations.pocketbase_admin_email")
    if not ADMIN_PASSWORD:
        missing.append("integrations.pocketbase_admin_password")
    if missing:
        print(f"ERROR: Missing required integration settings: {', '.join(missing)}")
        print("Set them in data/user/settings/integrations.json before running this script.")
        sys.exit(1)


def _get_client():
    try:
        from pocketbase import PocketBase  # type: ignore[import]
    except ImportError:
        print("ERROR: pocketbase package not installed.")
        print("Run: pip install pocketbase")
        sys.exit(1)

    pb = PocketBase(POCKETBASE_BASE_URL)
    pb.admins.auth_with_password(ADMIN_EMAIL, ADMIN_PASSWORD)
    return pb


def _existing_collections(pb) -> set[str]:
    try:
        collections = pb.collections.get_full_list()
        return {c.name for c in collections}
    except Exception:
        return set()


def _create_if_missing(pb, name: str, schema: dict, existing: set[str]) -> bool:
    if name in existing:
        print(f"  skip  {name} (already exists)")
        return True
    try:
        pb.collections.create(schema)
        print(f"  create {name}")
        return True
    except Exception as exc:
        # PocketBase renamed the collection field list from ``schema`` to
        # ``fields``. Retry the same definition in the modern shape.
        if "schema" in schema:
            modern = dict(schema)
            modern["fields"] = modern.pop("schema")
            try:
                pb.collections.create(modern)
                print(f"  create {name}")
                return True
            except Exception as modern_exc:
                print(f"  ERROR creating {name}: {modern_exc}")
                return False
        print(f"  ERROR creating {name}: {exc}")
        return False


def _field_for_api(field: dict) -> dict:
    """Undo pocketbase-python's key normalization before a schema PATCH."""
    payload = dict(field)
    if "auto_generate_pattern" in payload:
        payload["autogeneratePattern"] = payload.pop("auto_generate_pattern")
    if "primary_key" in payload:
        payload["primaryKey"] = payload.pop("primary_key")
    return payload


def _ensure_fields(pb, collection_name: str, required_fields: list[dict]) -> bool:
    """Add missing fields without replacing data in an existing collection.

    PocketBase renamed the collection payload from ``schema`` to ``fields``.
    Supporting both shapes lets operators safely rerun this script during an
    upgrade instead of manually editing production collections.
    """
    try:
        collection = next(
            item for item in pb.collections.get_full_list() if item.name == collection_name
        )
        current_fields = getattr(collection, "fields", None)
        payload_key = "fields"
        if current_fields is None:
            current_fields = getattr(collection, "schema", [])
            payload_key = "schema"
        existing_names = {
            (field.get("name") if isinstance(field, dict) else getattr(field, "name", ""))
            for field in current_fields
        }
        missing = [field for field in required_fields if field["name"] not in existing_names]
        if not missing:
            return True
        preserved = [
            _field_for_api(field) if isinstance(field, dict) else vars(field)
            for field in current_fields
        ]
        pb.collections.update(
            collection.id,
            {payload_key: [*preserved, *missing]},
        )
        print(f"  update {collection_name} (added {', '.join(field['name'] for field in missing)})")
        return True
    except Exception as exc:
        print(f"  ERROR updating {collection_name}: {exc}")
        return False


def main():
    _require_env()
    print(f"Connecting to PocketBase at {POCKETBASE_BASE_URL} ...")
    pb = _get_client()
    print("Authenticated as admin.")

    existing = _existing_collections(pb)
    print(f"Found {len(existing)} existing collection(s): {sorted(existing) or '(none)'}\n")

    # Access control is enforced in the application layer, not by PocketBase
    # collection rules: the backend connects with a single admin-authenticated
    # client (see services/pocketbase_client.py), which bypasses collection
    # RBAC entirely, so the rules below stay empty by design. Per-user session
    # isolation is implemented in PocketBaseSessionStore by stamping every
    # session row with ``user_id`` and filtering every query by the current
    # user. Do NOT rely on these listRule/viewRule strings for isolation.
    collections = [
        # ----------------------------------------------------------------
        # sessions  (``user_id`` populated + filtered by PocketBaseSessionStore)
        # ----------------------------------------------------------------
        {
            "name": "sessions",
            "type": "base",
            "schema": [
                {"name": "session_id", "type": "text", "required": True},
                {"name": "user_id", "type": "text", "required": False},
                {"name": "title", "type": "text", "required": False},
                {"name": "compressed_summary", "type": "text", "required": False},
                {"name": "summary_up_to_msg_id", "type": "number", "required": False},
                {"name": "preferences_json", "type": "json", "required": False},
                {"name": "folder_id", "type": "text", "required": False},
                {"name": "session_activity_at", "type": "number", "required": False},
                {"name": "capability", "type": "text", "required": False},
                {"name": "status", "type": "text", "required": False},
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
        # ----------------------------------------------------------------
        # session_folders (one level only; ownership enforced by the store)
        # ----------------------------------------------------------------
        {
            "name": "session_folders",
            "type": "base",
            "schema": [
                {"name": "folder_id", "type": "text", "required": True},
                {"name": "user_id", "type": "text", "required": True},
                {"name": "name", "type": "text", "required": True},
                {"name": "folder_created_at", "type": "number", "required": True},
                {"name": "folder_updated_at", "type": "number", "required": True},
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
        # ----------------------------------------------------------------
        # messages
        # ----------------------------------------------------------------
        {
            "name": "messages",
            "type": "base",
            "schema": [
                {"name": "session_id", "type": "text", "required": True},
                {"name": "role", "type": "text", "required": True},
                {"name": "content", "type": "text", "required": False},
                {"name": "capability", "type": "text", "required": False},
                {"name": "events_json", "type": "json", "required": False},
                {"name": "attachments_json", "type": "json", "required": False},
                {"name": "msg_created_at", "type": "number", "required": False},
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
        # ----------------------------------------------------------------
        # turns
        # ----------------------------------------------------------------
        {
            "name": "turns",
            "type": "base",
            "schema": [
                {"name": "turn_id", "type": "text", "required": True},
                {"name": "session_id", "type": "text", "required": True},
                {"name": "capability", "type": "text", "required": False},
                {"name": "status", "type": "text", "required": False},
                {"name": "error", "type": "text", "required": False},
                {"name": "turn_created_at", "type": "number", "required": False},
                {"name": "turn_updated_at", "type": "number", "required": False},
                {"name": "finished_at", "type": "number", "required": False},
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
        # ----------------------------------------------------------------
        # turn_events
        # ----------------------------------------------------------------
        {
            "name": "turn_events",
            "type": "base",
            "schema": [
                {"name": "turn_id", "type": "text", "required": True},
                {"name": "session_id", "type": "text", "required": False},
                {"name": "seq", "type": "number", "required": True},
                {"name": "type", "type": "text", "required": False},
                {"name": "source", "type": "text", "required": False},
                {"name": "stage", "type": "text", "required": False},
                {"name": "content", "type": "text", "required": False},
                {"name": "metadata_json", "type": "json", "required": False},
                {"name": "event_timestamp", "type": "number", "required": False},
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
        # ----------------------------------------------------------------
        # knowledge_bases
        # ----------------------------------------------------------------
        {
            "name": "knowledge_bases",
            "type": "base",
            "schema": [
                {"name": "kb_name", "type": "text", "required": True},
                {"name": "user_id", "type": "text", "required": False},
                {"name": "description", "type": "text", "required": False},
                {"name": "rag_provider", "type": "text", "required": False},
                {"name": "needs_reindex", "type": "bool", "required": False},
                {"name": "status", "type": "text", "required": False},
                {"name": "kb_created_at", "type": "text", "required": False},
                {
                    "name": "raw_files",
                    "type": "file",
                    "required": False,
                    "options": {"maxSelect": 99, "maxSize": 52428800},
                },
            ],
            "listRule": "",
            "viewRule": "",
            "createRule": "",
            "updateRule": "",
            "deleteRule": "",
        },
    ]

    print("Creating collections:")
    setup_ok = True
    for col in collections:
        if not _create_if_missing(pb, col["name"], col, existing):
            setup_ok = False

    # ``_create_if_missing`` intentionally leaves existing collections alone;
    # explicitly migrate the session fields required by folder organization.
    if not _ensure_fields(
        pb,
        "sessions",
        [
            {"name": "folder_id", "type": "text", "required": False},
            {"name": "session_activity_at", "type": "number", "required": False},
        ],
    ):
        setup_ok = False
    if not _ensure_fields(
        pb,
        "session_folders",
        [
            {"name": "folder_id", "type": "text", "required": True},
            {"name": "user_id", "type": "text", "required": True},
            {"name": "name", "type": "text", "required": True},
            {"name": "folder_created_at", "type": "number", "required": True},
            {"name": "folder_updated_at", "type": "number", "required": True},
        ],
    ):
        setup_ok = False

    if not setup_ok:
        print("\nERROR: PocketBase setup did not complete; see errors above.")
        sys.exit(1)

    print("\nDone. PocketBase collections are ready.")
    print(f"Open the admin panel at {POCKETBASE_BASE_URL}/_/ to view and configure collections.")


if __name__ == "__main__":
    main()
