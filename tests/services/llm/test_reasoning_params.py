"""Tests for the centralized reasoning-effort registry."""

from __future__ import annotations

import pytest

from deeptutor.services.llm.reasoning_params import (
    build_openai_compatible_reasoning_body,
    build_openai_compatible_reasoning_kwargs,
    default_reasoning_effort_for,
    is_toggle_effort,
)
from deeptutor.services.provider_registry import find_by_name


class TestDefaultReasoningEffortFor:
    """Single source of truth for the implicit per-provider/model effort."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gemini-2.5-flash", "none"),
            ("gemini-2.5-flash-lite", "none"),
            ("GEMINI-2.5-FLASH", "none"),
            ("models/gemini-2.5-flash", "none"),
            ("gemini-2.5-pro", "minimal"),
            ("gemini-3.0-pro", "minimal"),
            ("gemini-3.6-flash", "minimal"),
            ("gemini-3.6-flash-latest", "minimal"),
            ("models/gemini-3.6-flash", "minimal"),
        ],
    )
    def test_gemini_thinking_models_default(self, model: str, expected: str) -> None:
        assert default_reasoning_effort_for("gemini", model) == expected

    @pytest.mark.parametrize(
        "model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    )
    def test_gemini_legacy_models_unaffected(self, model: str) -> None:
        assert default_reasoning_effort_for("gemini", model) is None

    def test_other_providers_unaffected(self) -> None:
        assert default_reasoning_effort_for("openai", "gpt-5") is None
        assert default_reasoning_effort_for("deepseek", "deepseek-v4") is None
        assert default_reasoning_effort_for("dashscope", "qwen3-max") is None

    def test_missing_provider_or_model(self) -> None:
        assert default_reasoning_effort_for(None, "gemini-2.5-flash") is None
        assert default_reasoning_effort_for("gemini", None) is None
        assert default_reasoning_effort_for("", "") is None

    def test_provider_name_case_insensitive(self) -> None:
        assert default_reasoning_effort_for("Gemini", "gemini-2.5-flash") == "none"
        assert default_reasoning_effort_for("GEMINI", "gemini-2.5-flash") == "none"


class TestBuildOpenAICompatibleReasoningKwargsForGemini:
    """The OpenAI-compat helper consults the same registry."""

    def test_gemini_25_defaults_off_when_unspecified(self) -> None:
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=None, binding="gemini", model="gemini-2.5-flash", reasoning_effort=None
        )
        assert kwargs == {"reasoning_effort": "none"}

    def test_models_prefix_still_matches(self) -> None:
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=None,
            binding="gemini",
            model="models/gemini-2.5-flash",
            reasoning_effort=None,
        )
        assert kwargs == {"reasoning_effort": "none"}

    def test_explicit_effort_takes_precedence(self) -> None:
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=None,
            binding="gemini",
            model="gemini-2.5-flash",
            reasoning_effort="high",
        )
        assert kwargs == {"reasoning_effort": "high"}

    def test_gemini_15_left_untouched(self) -> None:
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=None,
            binding="gemini",
            model="gemini-1.5-flash",
            reasoning_effort=None,
        )
        assert kwargs == {}

    def test_openai_left_untouched(self) -> None:
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=None, binding="openai", model="gpt-4o", reasoning_effort=None
        )
        assert kwargs == {}


class TestToggleEffortValues:
    """``"on"``/``"off"`` are UI thinking-toggle values — never sent top-level."""

    def test_on_off_ignored_without_thinking_style(self) -> None:
        for effort in ("on", "off"):
            kwargs = build_openai_compatible_reasoning_kwargs(
                spec=None, binding="openai", model="gpt-4o", reasoning_effort=effort
            )
            assert kwargs == {}

    def test_on_off_drive_enable_thinking_providers(self) -> None:
        spec = find_by_name("dashscope")

        assert spec is not None
        off = build_openai_compatible_reasoning_kwargs(
            spec=spec, binding="dashscope", model="qwen3.7-flash", reasoning_effort="off"
        )
        on = build_openai_compatible_reasoning_kwargs(
            spec=spec, binding="dashscope", model="qwen3.7-flash", reasoning_effort="on"
        )
        assert off == {"extra_body": {"enable_thinking": False}}
        assert on == {"extra_body": {"enable_thinking": True}}

    def test_on_off_drive_thinking_type_providers_without_top_level(self) -> None:
        spec = find_by_name("deepseek")

        assert spec is not None
        off = build_openai_compatible_reasoning_kwargs(
            spec=spec, binding="deepseek", model="deepseek-v4-flash", reasoning_effort="off"
        )
        on = build_openai_compatible_reasoning_kwargs(
            spec=spec, binding="deepseek", model="deepseek-v4-flash", reasoning_effort="on"
        )
        assert off == {"extra_body": {"thinking": {"type": "disabled"}}}
        assert on == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_minimal_legacy_disable_semantics_unchanged(self) -> None:
        spec = find_by_name("dashscope")

        assert spec is not None
        kwargs = build_openai_compatible_reasoning_kwargs(
            spec=spec, binding="dashscope", model="qwen3.7-flash", reasoning_effort="minimal"
        )
        assert kwargs == {"extra_body": {"enable_thinking": False}}

    def test_levels_pass_through_verbatim(self) -> None:
        spec = find_by_name("kimi_coding_plan")

        assert spec is not None
        for effort in ("low", "high", "max"):
            kwargs = build_openai_compatible_reasoning_kwargs(
                spec=spec,
                binding="kimi_coding_plan",
                model="kimi-k3",
                reasoning_effort=effort,
            )
            assert kwargs == {"reasoning_effort": effort}

    def test_is_toggle_effort(self) -> None:
        assert is_toggle_effort("on")
        assert is_toggle_effort("OFF")
        assert not is_toggle_effort("high")
        assert not is_toggle_effort("")
        assert not is_toggle_effort(None)


def test_openrouter_reasoning_uses_normalized_nested_body() -> None:
    spec = find_by_name("openrouter")

    sdk_kwargs = build_openai_compatible_reasoning_kwargs(
        spec=spec,
        binding="openrouter",
        model="openai/gpt-5.6-sol",
        reasoning_effort="max",
    )
    raw_body = build_openai_compatible_reasoning_body(
        spec=spec,
        binding="openrouter",
        model="openai/gpt-5.6-sol",
        reasoning_effort="max",
    )

    assert sdk_kwargs == {"extra_body": {"reasoning": {"effort": "max"}}}
    assert raw_body == {"reasoning": {"effort": "max"}}
