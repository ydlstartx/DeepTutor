"""IMA account credentials stay with the human account that owns each KB."""

from __future__ import annotations

import json

from deeptutor.services.rag.pipelines.ima.config import (
    ImaConfig,
    load_account_settings,
    resolve_kb_config,
    save_account_settings,
)


def test_admin_and_user_ima_settings_are_stored_separately(mu_isolated_root, as_user) -> None:
    with as_user("u_admin", role="admin"):
        save_account_settings({"client_id": "admin-client", "api_key": "admin-key"})

    with as_user("u_alice", role="user"):
        save_account_settings({"client_id": "alice-client", "api_key": "alice-key"})
        assert load_account_settings(include_process_overrides=False)["client_id"] == "alice-client"

    with as_user("u_admin", role="admin"):
        assert load_account_settings(include_process_overrides=False)["client_id"] == "admin-client"

    admin_file = mu_isolated_root / "data" / "user" / "settings" / "ima.json"
    alice_file = mu_isolated_root / "data" / "users" / "u_alice" / "user" / "settings" / "ima.json"
    assert json.loads(admin_file.read_text(encoding="utf-8"))["client_id"] == "admin-client"
    assert json.loads(alice_file.read_text(encoding="utf-8"))["client_id"] == "alice-client"


def test_deployment_ima_env_override_is_not_exposed_to_users(
    mu_isolated_root, as_user, monkeypatch
) -> None:
    monkeypatch.setenv("IMA_CLIENT_ID", "deployment-client")
    monkeypatch.setenv("IMA_API_KEY", "deployment-key")

    with as_user("u_admin", role="admin"):
        assert load_account_settings()["client_id"] == "deployment-client"

    with as_user("u_alice", role="user"):
        settings = load_account_settings()
        assert settings["client_id"] == ""
        assert settings["api_key"] == ""


def test_kb_config_fallback_uses_the_kb_owner_not_the_requester(mu_isolated_root, as_user) -> None:
    admin_base = mu_isolated_root / "data" / "knowledge_bases"
    alice_base = mu_isolated_root / "data" / "users" / "u_alice" / "knowledge_bases"
    partner_base = (
        mu_isolated_root / "data" / "partners" / "study-bot" / "workspace" / "knowledge_bases"
    )

    with as_user("u_admin", role="admin"):
        save_account_settings({"client_id": "admin-client", "api_key": "admin-key"})
    with as_user("u_alice", role="user"):
        save_account_settings({"client_id": "alice-client", "api_key": "alice-key"})

        admin_config = resolve_kb_config(
            {"knowledge_base_id": "admin-kb"},
            kb_base_dir=admin_base,
        )
        own_config = resolve_kb_config(
            {"knowledge_base_id": "alice-kb"},
            kb_base_dir=alice_base,
        )
        partner_config = resolve_kb_config(
            {"knowledge_base_id": "partner-kb"},
            kb_base_dir=partner_base,
        )
        pinned_config = resolve_kb_config(
            {
                "client_id": "pinned-client",
                "api_key": "pinned-key",
                "knowledge_base_id": "pinned-kb",
            },
            kb_base_dir=admin_base,
        )

    assert admin_config == ImaConfig("admin-client", "admin-key", "admin-kb")
    assert own_config == ImaConfig("alice-client", "alice-key", "alice-kb")
    assert partner_config == ImaConfig("admin-client", "admin-key", "partner-kb")
    assert pinned_config == ImaConfig("pinned-client", "pinned-key", "pinned-kb")
