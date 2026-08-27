"""Aliyun DashScope embedding adapter (text + multimodal).

Uses the ``dashscope`` Python SDK rather than the OpenAI contract because
DashScope's native API shape does not match it. DashScope splits embeddings
across two surfaces served from different endpoints, and the SDK derives the
endpoint from the model id:

* multimodal models (``qwen3-vl-embedding``, ``multimodal-embedding-v1``) use
  ``dashscope.MultiModalEmbedding.call`` (``input=[{text|image|video}]`` +
  ``parameters={dimension, enable_fusion}``);
* text models (``text-embedding-v1..v4``) use ``dashscope.TextEmbedding.call``
  (``input=[str, ...]``).

Routing a text model through the multimodal call sends it to the multimodal
endpoint and fails with HTTP 400 "url error" (issue #660), so ``embed`` picks
the surface from the model id. Both calls are synchronous, so we run them in a
thread pool to keep the embedding stack non-blocking.
"""

from __future__ import annotations

import asyncio
from http import HTTPStatus
import logging
from typing import Any, Dict, List

from deeptutor.services.config.embedding_endpoint import (
    is_dashscope_multimodal_embedding_model,
)

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class DashScopeMultiModalEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Aliyun DashScope (Bailian) text + multimodal embedding."""

    MODELS_INFO = {
        "qwen3-vl-embedding": {
            "default": 2560,
            "dimensions": [256, 512, 768, 1024, 1536, 2048, 2560],
            "multimodal": True,
            # DashScope rejects image batches above 10 even though its text
            # embedding endpoint accepts a larger provider-level batch.
            "max_multimodal_batch_items": 10,
        },
        "multimodal-embedding-v1": {
            "default": 1536,
            "dimensions": [],
            "multimodal": True,
        },
        "text-embedding-v3": {
            "default": 1024,
            "dimensions": [],
            "multimodal": False,
        },
        "text-embedding-v4": {
            "default": 1024,
            "dimensions": [],
            "multimodal": False,
        },
    }

    _MAX_RETRIES = 5
    _RETRY_BACKOFF = 1.0
    _RATE_LIMIT_BACKOFF = 5.0

    @staticmethod
    def _is_retryable(status_code: Any, code: str) -> bool:
        """429 / ``Throttling.*`` and 5xx deserve another attempt; other 4xx are permanent."""
        if status_code == 429 or code.startswith("Throttling"):
            return True
        try:
            return int(status_code) >= 500
        except (TypeError, ValueError):
            return False

    async def _call_sdk_with_retry(self, sdk_call: Any, model_name: str, **kwargs: Any) -> Any:
        """Run one synchronous DashScope SDK call with throttling/transient retry.

        Mirrors the openai_compatible adapter's policy: 429/Throttling and 5xx
        responses plus SDK/network-level exceptions are retried with exponential
        backoff; permanent 4xx fails immediately. DashScope responses carry no
        ``Retry-After`` header, so backoff is purely exponential. Account rate
        quotas (``Throttling.RateQuota``) are routine while indexing with
        several concurrent embedding workers, and without this a single 429
        fails the whole document.
        """
        resp: Any = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                resp = await asyncio.to_thread(sdk_call, **kwargs)
            except Exception as exc:
                # Network/SDK-level failure (connection reset, read timeout, …).
                if attempt >= self._MAX_RETRIES:
                    logger.error(
                        f"DashScope embedding call failed after {1 + self._MAX_RETRIES} attempts "
                        f"({type(exc).__name__}: {exc})"
                    )
                    raise
                wait = self._RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    f"DashScope embedding call error ({type(exc).__name__}: {exc}) "
                    f"on attempt {attempt + 1}/{1 + self._MAX_RETRIES}, retrying in {wait:.1f}s..."
                )
                await asyncio.sleep(wait)
                continue

            status_code = getattr(resp, "status_code", None)
            code = str(getattr(resp, "code", "") or "")
            if status_code in (None, HTTPStatus.OK) or not self._is_retryable(status_code, code):
                return resp
            if attempt >= self._MAX_RETRIES:
                return resp  # retries exhausted — _raise_on_error surfaces the diagnostics
            rate_limited = status_code == 429 or code.startswith("Throttling")
            base = self._RATE_LIMIT_BACKOFF if rate_limited else self._RETRY_BACKOFF
            wait = base * (2**attempt)
            logger.warning(
                f"DashScope embedding call retryable failure (status={status_code}, "
                f"code={code}, model={model_name}) on attempt {attempt + 1}/"
                f"{1 + self._MAX_RETRIES}, retrying in {wait:.1f}s..."
            )
            await asyncio.sleep(wait)
        return resp

    def _build_contents(self, request: EmbeddingRequest) -> List[Dict[str, Any]]:
        if request.contents:
            return [item for item in request.contents if isinstance(item, dict)]
        return [{"text": text} for text in request.texts]

    def _build_parameters(self, request: EmbeddingRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        dim_value = request.dimensions or self.dimensions
        if dim_value:
            params["dimension"] = dim_value
        if request.enable_fusion is not None:
            params["enable_fusion"] = bool(request.enable_fusion)
        return params

    def _build_text_inputs(self, request: EmbeddingRequest) -> List[str]:
        """Flatten the request to the plain string list TextEmbedding expects."""
        if request.texts:
            return list(request.texts)
        return [
            item["text"]
            for item in (request.contents or [])
            if isinstance(item, dict) and item.get("text")
        ]

    def _build_text_parameters(self, request: EmbeddingRequest) -> Dict[str, Any]:
        # TextEmbedding takes `dimension` (v3/v4 support it) but has no
        # `enable_fusion` — that is a multimodal-only knob.
        params: Dict[str, Any] = {}
        dim_value = request.dimensions or self.dimensions
        if dim_value:
            params["dimension"] = dim_value
        return params

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_name = request.model or self.model
        if is_dashscope_multimodal_embedding_model(model_name):
            return await self._embed_multimodal(request, model_name)
        return await self._embed_text(request, model_name)

    async def _embed_multimodal(
        self, request: EmbeddingRequest, model_name: str
    ) -> EmbeddingResponse:
        try:
            from dashscope import MultiModalEmbedding
        except ImportError as exc:
            raise ImportError(
                "dashscope SDK not installed. Run `pip install dashscope` "
                "(or add to your project deps) to enable Aliyun DashScope."
            ) from exc

        contents = self._build_contents(request)
        parameters = self._build_parameters(request)

        logger.debug(
            "Calling dashscope.MultiModalEmbedding.call "
            f"(model={model_name}, items={len(contents)}, params={parameters})"
        )

        # SDK call is sync — run in worker thread to avoid blocking the loop.
        # IMPORTANT: the dashscope SDK takes a flat list for `input`
        # (e.g. ``input=[{"text": "..."}]``) and internally wraps it as
        # ``{"contents": [...]}`` before POSTing to the REST endpoint. Do NOT
        # pass ``{"contents": contents}`` here — that produces a double-wrap
        # and the API responds with HTTP 400 ("Input should be a valid list").
        resp = await self._call_sdk_with_retry(
            MultiModalEmbedding.call,
            model_name,
            api_key=self.api_key,
            model=model_name,
            input=contents,
            **parameters,
        )

        self._raise_on_error(resp, model_name)
        return self._parse_response(resp, model_name, request)

    async def _embed_text(self, request: EmbeddingRequest, model_name: str) -> EmbeddingResponse:
        try:
            from dashscope import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "dashscope SDK not installed. Run `pip install dashscope` "
                "(or add to your project deps) to enable Aliyun DashScope."
            ) from exc

        inputs = self._build_text_inputs(request)
        parameters = self._build_text_parameters(request)

        logger.debug(
            "Calling dashscope.TextEmbedding.call "
            f"(model={model_name}, items={len(inputs)}, params={parameters})"
        )

        # TextEmbedding.call POSTs to the DashScope text-embedding endpoint and
        # accepts a flat list of strings for `input`. Response/usage/error shape
        # matches MultiModalEmbedding, so we reuse the shared parsers below.
        resp = await self._call_sdk_with_retry(
            TextEmbedding.call,
            model_name,
            api_key=self.api_key,
            model=model_name,
            input=inputs,
            **parameters,
        )

        self._raise_on_error(resp, model_name)
        return self._parse_response(resp, model_name, request)

    def _raise_on_error(self, resp: Any, model_name: str) -> None:
        status_code = getattr(resp, "status_code", None)
        if status_code is None or status_code == HTTPStatus.OK:
            return
        code = getattr(resp, "code", "") or ""
        message = getattr(resp, "message", "") or ""
        request_id = getattr(resp, "request_id", "") or ""
        raise RuntimeError(
            f"DashScope MultiModalEmbedding call failed: "
            f"status={status_code}, code={code}, message={message}, "
            f"model={model_name}, request_id={request_id}"
        )

    def _parse_response(
        self, resp: Any, model_name: str, request: EmbeddingRequest
    ) -> EmbeddingResponse:
        output = getattr(resp, "output", None)
        if output is None:
            raise ValueError(
                f"DashScope response missing `output` (request_id={getattr(resp, 'request_id', '')})"
            )

        # `output` is dict-like in the SDK.
        if isinstance(output, dict):
            raw = output.get("embeddings") or []
        else:
            raw = getattr(output, "embeddings", None) or []

        embeddings: List[List[float]] = []
        for item in raw:
            if isinstance(item, dict):
                vec = item.get("embedding")
            else:
                vec = getattr(item, "embedding", None)
            if vec is None:
                continue
            embeddings.append(list(vec))

        if not embeddings:
            raise ValueError(
                "DashScope response parsed successfully but no embedding vectors were returned."
            )

        usage = getattr(resp, "usage", {}) or {}
        if not isinstance(usage, dict):
            usage = {
                k: getattr(usage, k, None)
                for k in ("input_tokens", "output_tokens", "total_tokens")
                if hasattr(usage, k)
            }

        actual_dims = len(embeddings[0]) if embeddings else 0
        logger.info(
            f"Successfully generated {len(embeddings)} DashScope embeddings "
            f"(model: {model_name}, dimensions: {actual_dims}, "
            f"fusion={request.enable_fusion})"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            dimensions=actual_dims,
            usage=usage,
        )

    def get_model_info(self) -> Dict[str, Any]:
        info = self.MODELS_INFO.get(self.model or "", {})
        return {
            "model": self.model,
            "dimensions": info.get("default", self.dimensions),
            "supported_dimensions": info.get("dimensions", []),
            "supports_variable_dimensions": bool(info.get("dimensions")),
            "multimodal": bool(info.get("multimodal", False)),
            "max_multimodal_batch_items": info.get("max_multimodal_batch_items"),
            "provider": "aliyun",
        }
