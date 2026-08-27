"""Password self-service and admin-reset contracts for built-in multi-user auth."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@pytest.fixture
def password_client(mu_isolated_root, monkeypatch):
    import deeptutor.api.routers.auth as auth_router
    from deeptutor.multi_user.identity import save_user
    from deeptutor.services import auth as auth_service

    monkeypatch.setattr(auth_service, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_service, "POCKETBASE_ENABLED", False)
    monkeypatch.setattr(auth_service, "AUTH_SECRET", "password-test-secret")
    monkeypatch.setattr(auth_router, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth_router, "POCKETBASE_ENABLED", False)

    alice = save_user(
        "alice",
        auth_service.hash_password("alice-password-1234"),
        role="admin",
    )
    bob = save_user(
        "bob",
        auth_service.hash_password("bob-password-1234"),
        role="user",
    )
    tokens = {
        "alice": auth_service.create_token("alice", "admin", alice["id"]),
        "bob": auth_service.create_token("bob", "user", bob["id"]),
    }

    app = FastAPI()
    app.include_router(auth_router.router, prefix="/api/v1/auth")
    return TestClient(app), tokens


def test_user_changes_own_password_and_rotates_session(password_client):
    from deeptutor.multi_user.identity import load_users
    from deeptutor.services import auth as auth_service

    client, tokens = password_client
    response = client.put(
        "/api/v1/auth/profile/password",
        headers=_auth(tokens["bob"]),
        json={
            "current_password": "bob-password-1234",
            "new_password": "bob-password-5678",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert "dt_token=" in response.headers["set-cookie"]
    assert auth_service.authenticate("bob", "bob-password-1234") is None
    assert auth_service.authenticate("bob", "bob-password-5678") is not None
    assert auth_service.decode_token(tokens["bob"]) is None
    assert auth_service.decode_token(client.cookies["dt_token"]) is not None
    assert load_users()["bob"]["auth_version"] == 1


def test_user_password_change_rejects_wrong_current_same_and_short_passwords(password_client):
    from deeptutor.services import auth as auth_service

    client, tokens = password_client
    wrong = client.put(
        "/api/v1/auth/profile/password",
        headers=_auth(tokens["bob"]),
        json={"current_password": "wrong-password", "new_password": "new-password-1234"},
    )
    same = client.put(
        "/api/v1/auth/profile/password",
        headers=_auth(tokens["bob"]),
        json={
            "current_password": "bob-password-1234",
            "new_password": "bob-password-1234",
        },
    )
    short = client.put(
        "/api/v1/auth/profile/password",
        headers=_auth(tokens["bob"]),
        json={"current_password": "bob-password-1234", "new_password": "short"},
    )

    assert wrong.status_code == 400
    assert wrong.json()["detail"] == "Current password is incorrect"
    assert same.status_code == 400
    assert "must be different" in same.json()["detail"]
    assert short.status_code == 422
    assert auth_service.authenticate("bob", "bob-password-1234") is not None
    assert auth_service.decode_token(tokens["bob"]) is not None


def test_admin_resets_another_users_password_and_revokes_their_sessions(password_client):
    from deeptutor.services import auth as auth_service

    client, tokens = password_client
    response = client.put(
        "/api/v1/auth/users/bob/password",
        headers=_auth(tokens["alice"]),
        json={"new_password": "admin-reset-1234"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "username": "bob"}
    assert auth_service.authenticate("bob", "bob-password-1234") is None
    assert auth_service.authenticate("bob", "admin-reset-1234") is not None
    assert auth_service.decode_token(tokens["bob"]) is None
    assert auth_service.decode_token(tokens["alice"]) is not None


def test_admin_password_reset_enforces_auth_ownership_and_existence(password_client):
    client, tokens = password_client
    anonymous = client.put(
        "/api/v1/auth/users/bob/password",
        json={"new_password": "new-password-1234"},
    )
    regular_user = client.put(
        "/api/v1/auth/users/alice/password",
        headers=_auth(tokens["bob"]),
        json={"new_password": "new-password-1234"},
    )
    own_account = client.put(
        "/api/v1/auth/users/alice/password",
        headers=_auth(tokens["alice"]),
        json={"new_password": "new-password-1234"},
    )
    missing = client.put(
        "/api/v1/auth/users/nobody/password",
        headers=_auth(tokens["alice"]),
        json={"new_password": "new-password-1234"},
    )

    assert anonymous.status_code == 401
    assert regular_user.status_code == 403
    assert own_account.status_code == 400
    assert own_account.json()["detail"] == "Use your profile to change your own password"
    assert missing.status_code == 404


def test_password_endpoints_explain_pocketbase_boundary(password_client, monkeypatch):
    import deeptutor.api.routers.auth as auth_router

    client, tokens = password_client
    monkeypatch.setattr(auth_router, "POCKETBASE_ENABLED", True)

    own = client.put(
        "/api/v1/auth/profile/password",
        headers=_auth(tokens["bob"]),
        json={
            "current_password": "bob-password-1234",
            "new_password": "bob-password-5678",
        },
    )
    reset = client.put(
        "/api/v1/auth/users/bob/password",
        headers=_auth(tokens["alice"]),
        json={"new_password": "admin-reset-1234"},
    )

    assert own.status_code == 400
    assert "PocketBase" in own.json()["detail"]
    assert reset.status_code == 400
    assert "PocketBase" in reset.json()["detail"]


def test_bootstrap_admin_password_change_adopts_account(mu_isolated_root, monkeypatch):
    from deeptutor.multi_user.identity import USERS_FILE, load_users
    from deeptutor.services import auth as auth_service

    env_admin = "operator"
    monkeypatch.setattr(auth_service, "AUTH_USERNAME", env_admin)
    monkeypatch.setattr(
        auth_service,
        "AUTH_PASSWORD_HASH",
        auth_service.hash_password("bootstrap-pass-1234"),
    )
    assert auth_service.verify_user_password(env_admin, "bootstrap-pass-1234")
    assert auth_service.set_user_password(env_admin, "operator-password-5678")

    stored = load_users()[env_admin]
    assert USERS_FILE.exists()
    assert stored["role"] == "admin"
    assert stored["auth_version"] == 1
    assert not auth_service.verify_user_password(env_admin, "bootstrap-pass-1234")
    assert auth_service.verify_user_password(env_admin, "operator-password-5678")
