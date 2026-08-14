"""Wall-clock attempt timeout in ``provider_core.base.LLMProvider``.

SDK/socket read timeouts only fire on byte-level silence, so a proxy that
trickles keep-alive bytes can hold a request open indefinitely (observed:
an 80+ minute stall during KB indexing). The attempt cap bounds the whole
call and routes the timeout into the normal transient-retry path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deeptutor.services.llm.provider_core.base import LLMProvider, LLMResponse


class _HangingProvider(LLMProvider):
    """Every attempt hangs forever; only the attempt cap can save it."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.attempts += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    async def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.attempts += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_hanging_attempt_times_out_and_retries(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S", "0.05")
    provider = _HangingProvider()

    # Outer guard: if the cap wiring regresses, fail in seconds instead of
    # hanging on the provider's 3600s sleep.
    resp = await asyncio.wait_for(
        provider.chat_with_retry(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            retry_delays=(0.01, 0.01),
        ),
        timeout=10,
    )

    assert resp.finish_reason == "error"
    assert "timed out" in (resp.content or "")
    assert provider.attempts == 3  # initial attempt + 2 retries, each capped


@pytest.mark.asyncio
async def test_stream_attempt_uses_stream_timeout(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_LLM_STREAM_ATTEMPT_TIMEOUT_S", "0.05")
    # The chat cap must not govern streaming calls.
    monkeypatch.setenv("DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S", "3600")
    provider = _HangingProvider()

    resp = await asyncio.wait_for(
        provider.chat_stream_with_retry(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            retry_delays=(),
        ),
        timeout=10,  # same fail-fast guard against a wiring regression
    )

    assert resp.finish_reason == "error"
    assert "timed out" in (resp.content or "")
    assert provider.attempts == 1


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_by_the_cap(monkeypatch) -> None:
    monkeypatch.setenv("DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S", "60")
    provider = _HangingProvider()

    task = asyncio.create_task(
        provider.chat_with_retry(
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            retry_delays=(0.01,),
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_attempt_timeout_env_override(monkeypatch) -> None:
    name = "DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S"
    monkeypatch.delenv(name, raising=False)
    assert LLMProvider._attempt_timeout(name, 900.0) == 900.0
    monkeypatch.setenv(name, "42")
    assert LLMProvider._attempt_timeout(name, 900.0) == 42.0
    # Non-positive disables the cap explicitly; unparsable keeps the default —
    # a typo must not silently switch off the safety net.
    monkeypatch.setenv(name, "0")
    assert LLMProvider._attempt_timeout(name, 900.0) == 0.0
    monkeypatch.setenv(name, "bogus")
    assert LLMProvider._attempt_timeout(name, 900.0) == 900.0
    # NaN compares False to everything (would disable); inf never elapses.
    monkeypatch.setenv(name, "nan")
    assert LLMProvider._attempt_timeout(name, 900.0) == 900.0
    monkeypatch.setenv(name, "inf")
    assert LLMProvider._attempt_timeout(name, 900.0) == 900.0


class _FallbackHangingProvider(LLMProvider):
    """Fails non-transiently when images are present, then hangs on the
    text-only Stage-2 fallback call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        from deeptutor.services.llm.multimodal import has_image_parts

        self.calls += 1
        if has_image_parts(messages):
            return LLMResponse(content="this model does not support images", finish_reason="error")
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")  # pragma: no cover

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_image_fallback_call_is_also_capped(monkeypatch) -> None:
    """The Stage-2 text-only retry is capped too, and a hang there comes back
    as the same normalized error response as a main-attempt timeout."""
    monkeypatch.setenv("DEEPTUTOR_LLM_ATTEMPT_TIMEOUT_S", "0.05")
    provider = _FallbackHangingProvider()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
            ],
        }
    ]

    resp = await asyncio.wait_for(  # fail-fast guard if the cap regresses
        provider.chat_with_retry(
            messages=messages, model="m", retry_delays=(), allow_image_fallback=True
        ),
        timeout=10,
    )
    assert resp.finish_reason == "error"
    assert "timed out" in (resp.content or "")
    assert provider.calls == 2  # initial (with images) + fallback (text-only)
