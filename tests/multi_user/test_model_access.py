"""Runtime resolution of administrator-granted model settings."""

from __future__ import annotations

from deeptutor.multi_user import model_access
from deeptutor.services.model_selection.runtime import resolve_llm_config_for_selection


def _admin_catalog(reasoning_effort: str) -> dict:
    return {
        "version": 1,
        "services": {
            "llm": {
                "active_profile_id": "openrouter-profile",
                "active_model_id": "gpt-56-sol",
                "profiles": [
                    {
                        "id": "openrouter-profile",
                        "name": "OpenRouter",
                        "binding": "openrouter",
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_key": "test-secret",
                        "models": [
                            {
                                "id": "gpt-56-sol",
                                "name": "GPT-5.6 Sol",
                                "model": "openai/gpt-5.6-sol",
                                "reasoning_effort": reasoning_effort,
                            }
                        ],
                    }
                ],
            }
        },
    }


def test_granted_user_resolves_admin_model_reasoning_effort(
    mu_isolated_root, as_user, monkeypatch
) -> None:
    del mu_isolated_root
    model_access.admin_catalog_service().save(_admin_catalog("high"))
    monkeypatch.setattr(
        model_access,
        "load_grant",
        lambda _user_id=None: {
            "models": {
                "llm": [
                    {
                        "profile_id": "openrouter-profile",
                        "model_ids": ["gpt-56-sol"],
                    }
                ]
            }
        },
    )

    selection = {"profile_id": "openrouter-profile", "model_id": "gpt-56-sol"}
    with as_user("u_alice"):
        assert model_access.apply_allowed_llm_selection(selection) == selection
        resolved = resolve_llm_config_for_selection(selection)

    assert resolved.model == "openai/gpt-5.6-sol"
    assert resolved.reasoning_effort == "high"
