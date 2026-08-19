"""Tests for the DashScope (Aliyun) MultiModalEmbedding adapter."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from deeptutor.services.embedding.adapters.base import EmbeddingRequest
from deeptutor.services.embedding.adapters.dashscope_native import (
    DashScopeMultiModalEmbeddingAdapter,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        output: dict | None = None,
        usage: dict | None = None,
        code: str = "",
        message: str = "",
        request_id: str = "req-1",
    ) -> None:
        self.status_code = status_code
        self.output = output if output is not None else {"embeddings": []}
        self.usage = usage or {}
        self.code = code
        self.message = message
        self.request_id = request_id


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> dict[str, Any]:
    """Stub both DashScope embedding surfaces and record which one is called."""
    captured: dict[str, Any] = {}

    def _surface(name: str):
        def fake_call(*, api_key: str, model: str, input: Any, **kwargs: Any) -> _FakeResponse:  # noqa: A002
            captured.update(
                surface=name,
                api_key=api_key,
                model=model,
                input=input,
                kwargs=kwargs,
            )
            return response

        return fake_call

    fake_module = types.SimpleNamespace(
        MultiModalEmbedding=types.SimpleNamespace(call=_surface("multimodal")),
        TextEmbedding=types.SimpleNamespace(call=_surface("text")),
    )
    monkeypatch.setitem(sys.modules, "dashscope", fake_module)
    return captured


@pytest.mark.asyncio
async def test_text_only_translates_texts_to_contents(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        output={"embeddings": [{"index": 0, "embedding": [0.1, 0.2, 0.3], "type": "vl"}]},
    )
    captured = _install_fake_sdk(monkeypatch, response)

    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk-dashscope",
            "base_url": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            "model": "qwen3-vl-embedding",
            "dimensions": 1024,
            "request_timeout": 5,
        }
    )
    resp = await adapter.embed(
        EmbeddingRequest(texts=["hello", "world"], model="qwen3-vl-embedding")
    )

    # SDK takes a flat list — it wraps as {"contents": ...} internally.
    assert captured["surface"] == "multimodal"
    assert captured["input"] == [{"text": "hello"}, {"text": "world"}]
    assert captured["model"] == "qwen3-vl-embedding"
    assert captured["api_key"] == "sk-dashscope"
    assert captured["kwargs"].get("dimension") == 1024
    assert "enable_fusion" not in captured["kwargs"]
    assert resp.embeddings == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_text_model_uses_text_embedding_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression for issue #660: a text model (text-embedding-v4) must go to the
    # TextEmbedding surface with a flat string list, NOT the multimodal endpoint
    # (which returns HTTP 400 "url error").
    response = _FakeResponse(
        output={"embeddings": [{"text_index": 0, "embedding": [0.1, 0.2]}]},
    )
    captured = _install_fake_sdk(monkeypatch, response)

    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk-dashscope",
            "base_url": "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
            "model": "text-embedding-v4",
            "dimensions": 1024,
            "request_timeout": 5,
        }
    )
    resp = await adapter.embed(
        EmbeddingRequest(texts=["hello", "world"], model="text-embedding-v4")
    )

    assert captured["surface"] == "text"
    assert captured["input"] == ["hello", "world"]  # flat strings, not {"text": ...}
    assert captured["model"] == "text-embedding-v4"
    assert captured["kwargs"].get("dimension") == 1024
    assert "enable_fusion" not in captured["kwargs"]
    assert resp.embeddings == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_multimodal_contents_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        output={"embeddings": [{"index": 0, "embedding": [0.4, 0.5], "type": "fusion"}]},
    )
    captured = _install_fake_sdk(monkeypatch, response)

    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk-dashscope",
            "base_url": "https://dashscope.aliyuncs.com/...",
            "model": "qwen3-vl-embedding",
            "dimensions": 0,
            "request_timeout": 5,
        }
    )
    contents = [{"text": "a slide"}, {"image": "https://example.com/img.png"}]
    resp = await adapter.embed(
        EmbeddingRequest(
            texts=[],
            model="qwen3-vl-embedding",
            contents=contents,
            enable_fusion=True,
        )
    )
    # SDK takes a flat list — it wraps as {"contents": ...} internally.
    assert captured["input"] == contents
    assert captured["kwargs"].get("enable_fusion") is True
    assert resp.embeddings == [[0.4, 0.5]]


@pytest.mark.asyncio
async def test_failure_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=400,
        output={"embeddings": []},
        code="InvalidParameter",
        message="dimension out of range",
    )
    _install_fake_sdk(monkeypatch, response)

    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk",
            "base_url": "https://dashscope.aliyuncs.com/...",
            "model": "qwen3-vl-embedding",
            "request_timeout": 5,
        }
    )
    with pytest.raises(RuntimeError) as ei:
        await adapter.embed(EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding"))
    assert "InvalidParameter" in str(ei.value)


def test_get_model_info_reports_multimodal_capability() -> None:
    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk",
            "base_url": "https://...",
            "model": "qwen3-vl-embedding",
        }
    )
    info = adapter.get_model_info()
    assert info["multimodal"] is True
    assert info["provider"] == "aliyun"
    assert 2560 in info["supported_dimensions"]


@pytest.mark.parametrize(
    ("model", "multimodal", "endpoint_tail"),
    [
        ("text-embedding-v4", False, "/text-embedding/text-embedding"),
        ("text-embedding-v3", False, "/text-embedding/text-embedding"),
        ("some-unknown-model", False, "/text-embedding/text-embedding"),
        ("qwen3-vl-embedding", True, "/multimodal-embedding/multimodal-embedding"),
        ("multimodal-embedding-v1", True, "/multimodal-embedding/multimodal-embedding"),
    ],
)
def test_dashscope_endpoint_routing(model: str, multimodal: bool, endpoint_tail: str) -> None:
    # Issue #660: the single source of truth that decides text vs multimodal
    # DashScope endpoint per model.
    from deeptutor.services.config.embedding_endpoint import (
        dashscope_embedding_endpoint,
        is_dashscope_multimodal_embedding_model,
    )

    assert is_dashscope_multimodal_embedding_model(model) is multimodal
    assert dashscope_embedding_endpoint(model).endswith(endpoint_tail)


# --------------------------------------------------------------------------- #
# Throttling / transient retry
# --------------------------------------------------------------------------- #


def _install_scripted_sdk(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> dict[str, int]:
    """Stub both DashScope surfaces to play back a scripted sequence.

    Each entry is either a ``_FakeResponse`` to return or an Exception to raise;
    the last entry repeats once the script is exhausted.
    """
    calls = {"count": 0}

    def fake_call(*, api_key: str, model: str, input: Any, **kwargs: Any) -> _FakeResponse:  # noqa: A002
        calls["count"] += 1
        outcome = script[min(calls["count"] - 1, len(script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake_module = types.SimpleNamespace(
        MultiModalEmbedding=types.SimpleNamespace(call=fake_call),
        TextEmbedding=types.SimpleNamespace(call=fake_call),
    )
    monkeypatch.setitem(sys.modules, "dashscope", fake_module)
    return calls


def _retrying_adapter() -> DashScopeMultiModalEmbeddingAdapter:
    adapter = DashScopeMultiModalEmbeddingAdapter(
        {
            "api_key": "sk",
            "base_url": "https://dashscope.aliyuncs.com/...",
            "model": "qwen3-vl-embedding",
            "request_timeout": 5,
        }
    )
    # Keep the retry tests instant.
    adapter._RATE_LIMIT_BACKOFF = 0.0
    adapter._RETRY_BACKOFF = 0.0
    return adapter


_OK = _FakeResponse(output={"embeddings": [{"index": 0, "embedding": [0.1], "type": "vl"}]})
_429 = _FakeResponse(status_code=429, code="Throttling.RateQuota", message="rate limit")


@pytest.mark.asyncio
async def test_throttled_call_is_retried_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_sdk(monkeypatch, [_429, _429, _OK])
    resp = await _retrying_adapter().embed(
        EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding")
    )
    assert resp.embeddings == [[0.1]]
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_throttling_exhaustion_raises_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_scripted_sdk(monkeypatch, [_429])
    with pytest.raises(RuntimeError) as ei:
        await _retrying_adapter().embed(EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding"))
    assert "Throttling.RateQuota" in str(ei.value)
    assert calls["count"] == 1 + DashScopeMultiModalEmbeddingAdapter._MAX_RETRIES


@pytest.mark.asyncio
async def test_server_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_sdk(
        monkeypatch,
        [_FakeResponse(status_code=503, code="ServiceUnavailable", message="busy"), _OK],
    )
    resp = await _retrying_adapter().embed(
        EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding")
    )
    assert resp.embeddings == [[0.1]]
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_permanent_4xx_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_sdk(
        monkeypatch,
        [_FakeResponse(status_code=400, code="InvalidParameter", message="bad input")],
    )
    with pytest.raises(RuntimeError):
        await _retrying_adapter().embed(EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding"))
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_sdk_exception_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_scripted_sdk(monkeypatch, [ConnectionError("reset by peer"), _OK])
    resp = await _retrying_adapter().embed(
        EmbeddingRequest(texts=["x"], model="qwen3-vl-embedding")
    )
    assert resp.embeddings == [[0.1]]
    assert calls["count"] == 2
