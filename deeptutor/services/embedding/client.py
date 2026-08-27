"""Unified embedding client backed by normalized provider runtime config."""

from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

from deeptutor.services.config.embedding_endpoint import (
    redact_embedding_endpoint_for_display,
)
from deeptutor.services.config.provider_runtime import (
    EMBEDDING_PROVIDERS,
    embedding_endpoint_validation_error,
)

from .adapters import ADAPTER_BACKENDS, BaseEmbeddingAdapter, EmbeddingRequest
from .config import EmbeddingConfig, get_embedding_config
from .validation import validate_embedding_batch

# Reusable executor for sync embedding calls made from inside a running event
# loop (embed_sync submits asyncio.run to a worker thread).
_sync_embed_executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


_BATCH_RANGE_PATTERN = re.compile(
    r"(?:image\s+)?batch(?:[_\s-]*size)?[^\n]{0,80}?"
    r"\[\s*\d+\s*,\s*(\d+)\s*\]",
    re.IGNORECASE,
)
_BATCH_MAX_PATTERN = re.compile(
    r"(?:image\s+)?batch(?:[_\s-]*size)?[^\n]{0,80}?"
    r"(?:max(?:imum)?|at\s+most|no\s+more\s+than)\D{0,16}(\d+)",
    re.IGNORECASE,
)
_HTTP_413_PATTERN = re.compile(
    r"(?:status(?:[_\s-]*code)?\s*[:=]\s*413\b|http(?:\s+status)?\s+413\b)",
    re.IGNORECASE,
)


def _batch_limit_from_error(exc: Exception) -> int | None:
    """Extract an explicit provider batch limit without adapting unrelated 4xx errors."""
    message = str(exc)
    for pattern in (_BATCH_RANGE_PATTERN, _BATCH_MAX_PATTERN):
        match = pattern.search(message)
        if match:
            limit = int(match.group(1))
            return limit if limit > 0 else None
    return None


def _is_adaptable_batch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if _batch_limit_from_error(exc) is not None:
        return True
    return (
        _HTTP_413_PATTERN.search(message) is not None
        or "payload too large" in message
        or "too many" in message
        and ("batch" in message or "input" in message)
    )


@contextmanager
def _sync_embed_executor() -> Iterator[concurrent.futures.ThreadPoolExecutor]:
    yield _sync_embed_executor_pool


def _resolve_adapter_class(binding: str) -> type[BaseEmbeddingAdapter]:
    provider = (binding or "").strip().lower()
    spec = EMBEDDING_PROVIDERS.get(provider)
    if spec is None:
        supported = sorted(EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding binding: '{binding}'. Supported: {', '.join(supported)}"
        )
    cls = ADAPTER_BACKENDS.get(spec.adapter)
    if cls is None:
        raise ValueError(
            f"No adapter registered for backend '{spec.adapter}' (binding='{binding}')"
        )
    return cls


class EmbeddingClient:
    """Unified embedding client for RAG and retrieval services."""

    # 全局发帖节流：KB reindex 时 LlamaIndex 用线程池并发调 embedding，
    # 每个线程经 _run_in_new_loop 跑独立 event loop——asyncio.Lock 绑定
    # 创建时的 loop，跨 loop 既不互斥还会挂死。必须用线程级锁。
    _spacing_lock: Any = None
    _last_request_monotonic: float = 0.0
    _thread_guard: Any = None

    @classmethod
    def _global_spacing_lock(cls):
        import threading

        if cls._spacing_lock is None:
            cls._thread_guard = threading.Lock()
            with cls._thread_guard:
                if cls._spacing_lock is None:
                    from threading import Lock as _TLock

                    cls._spacing_lock = _TLock()
        return cls._spacing_lock

    @staticmethod
    def _hold_spacing_lock():
        """Blocking acquire — cross-thread, loop-agnostic."""
        lock = EmbeddingClient._global_spacing_lock()
        lock.acquire()
        return lock

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_embedding_config()
        self.logger = logging.getLogger(__name__)
        # The client is recreated when its immutable model config changes, so
        # an instance-local learned cap is automatically model-specific.
        self._learned_multimodal_batch_size: int | None = None
        endpoint = self.config.effective_url or self.config.base_url
        problem = embedding_endpoint_validation_error(self.config.binding, endpoint)
        if problem:
            displayed_endpoint = redact_embedding_endpoint_for_display(endpoint)
            raise ValueError(
                f"{problem} Current Settings endpoint is {displayed_endpoint!r}. "
                "DeepTutor sends embedding requests to the Settings URL exactly; "
                "update the visible Endpoint URL instead of relying on hidden path appending."
            )
        adapter_class = _resolve_adapter_class(self.config.binding)
        self.adapter = adapter_class(
            {
                "api_key": self.config.api_key,
                "base_url": self.config.effective_url or self.config.base_url,
                "api_version": self.config.api_version,
                "model": self.config.model,
                "dimensions": self.config.dim,
                "send_dimensions": self.config.send_dimensions,
                "request_timeout": self.config.request_timeout,
                "extra_headers": self.config.extra_headers or {},
            }
        )
        self.logger.info(
            f"Initialized embedding client with {self.config.binding} adapter "
            f"(model: {self.config.model}, dimensions: {self.config.dim})"
        )

    async def embed(
        self,
        texts: List[str],
        progress_callback=None,
        *,
        input_type: str | None = None,
    ) -> List[List[float]]:
        """Embed text batches, optionally identifying their retrieval role."""
        if not texts:
            return []

        # Only adapters that opted in receive the role. Forwarding it to every
        # backend would change the request Jina has always sent (no `task`) and
        # silently invalidate the indexes built from it.
        role = input_type if getattr(self.adapter, "SUPPORTS_INPUT_TYPE", False) else None

        import asyncio

        # Clamp configured batch size against the provider's per-request item
        # cap. SiliconFlow Qwen3 family caps at 32; DashScope at 20; others
        # have generous defaults. Without this clamp, indexing a doc with many
        # chunks fails on the second batch even when "Test connection" passes.
        spec = EMBEDDING_PROVIDERS.get(self.config.binding)
        provider_max = spec.max_batch_items if spec else 256
        batch_size = max(1, min(self.config.batch_size, provider_max))
        if batch_size < self.config.batch_size:
            self.logger.info(
                f"Clamped batch_size {self.config.batch_size} -> {batch_size} "
                f"(provider '{self.config.binding}' max={provider_max})"
            )
        all_embeddings: List[List[float]] = []
        batch_delay = self.config.batch_delay
        expected_dim: int | None = None

        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i, start in enumerate(range(0, len(texts), batch_size)):
            batch = texts[start : start + batch_size]
            request = EmbeddingRequest(
                texts=batch,
                model=self.config.model,
                dimensions=self.config.dim or None,
                input_type=role,
            )
            try:
                # 全局发帖节流：线程级锁串行化"等待间隔+发帖"，跨线程/跨
                # event loop 互斥（asyncio 锁在新 loop 模型下失效的教训）。
                # asyncio.sleep 换成阻塞等待发帖侧可接受：DT 的 embedding
                # 并发都在后台线程里，阻塞不伤 API 主线程。
                import time as _time
                from time import monotonic as _mono

                _lock = EmbeddingClient._hold_spacing_lock()
                try:
                    delay = self.config.batch_delay
                    if delay > 0:
                        elapsed = _mono() - EmbeddingClient._last_request_monotonic
                        if elapsed < delay:
                            _time.sleep(delay - elapsed)
                    EmbeddingClient._last_request_monotonic = _mono()
                    response = await self.adapter.embed(request)
                finally:
                    _lock.release()
            except Exception as exc:
                # Capture batch context so the task log stream / KB diagnostics
                # show actionable info instead of a bare exception string.
                import traceback

                first_chunk_chars = len(batch[0]) if batch else 0
                longest_chunk_chars = max((len(t) for t in batch), default=0)
                self.logger.error(
                    f"Embedding batch failed "
                    f"(binding={self.config.binding}, model={self.config.model}, "
                    f"batch_index={i + 1}/{total_batches}, batch_items={len(batch)}, "
                    f"first_chunk_chars={first_chunk_chars}, "
                    f"longest_chunk_chars={longest_chunk_chars}): {exc}\n"
                    f"{traceback.format_exc()}"
                )
                raise
            validated = validate_embedding_batch(
                response.embeddings,
                expected_count=len(batch),
                binding=self.config.binding,
                model=self.config.model,
                batch_index=i + 1,
                total_batches=total_batches,
                start_index=start,
            )
            batch_dim = len(validated[0]) if validated else 0
            if expected_dim is None:
                expected_dim = batch_dim
            elif batch_dim != expected_dim:
                raise ValueError(
                    "Embedding provider returned inconsistent vector dimensions "
                    f"across batches (binding={self.config.binding}, "
                    f"model={self.config.model}): expected {expected_dim}, "
                    f"got {batch_dim} in batch {i + 1}/{total_batches}. "
                    "Use a single embedding model/dimension and re-index the knowledge base."
                )

            all_embeddings.extend(validated)

            # Report progress after each batch
            if progress_callback:
                try:
                    progress_callback(i + 1, total_batches)
                except Exception:
                    pass

            # Delay between batches to avoid rate limiting
            if i < total_batches - 1 and batch_delay > 0:
                await asyncio.sleep(batch_delay)

        self.logger.debug(
            f"Generated {len(all_embeddings)} embeddings using "
            f"{self.config.binding} (batch_size={batch_size})"
        )
        return all_embeddings

    def supports_multimodal_contents(self) -> bool:
        """Return whether the configured adapter/model accepts multimodal contents."""
        try:
            info = self.adapter.get_model_info()
            if "multimodal" in info:
                return bool(info.get("multimodal"))
        except Exception:
            pass

        spec = EMBEDDING_PROVIDERS.get(self.config.binding)
        return bool(spec and spec.multimodal)

    async def embed_contents(
        self,
        contents: List[Dict[str, Any]],
        *,
        progress_callback=None,
    ) -> List[List[float]]:
        """Embed provider-agnostic multimodal content items.

        ``contents`` uses the same simple contract as ``EmbeddingRequest``:
        ``[{"text": "..."}, {"image": "data:...|url"}, {"video": "..."}]``.
        """
        if not contents:
            return []
        if not self.supports_multimodal_contents():
            raise ValueError(
                "Configured embedding provider/model does not support multimodal contents."
            )

        import asyncio

        spec = EMBEDDING_PROVIDERS.get(self.config.binding)
        provider_max = spec.max_batch_items if spec else 256
        model_max: int | None = None
        try:
            raw_model_max = self.adapter.get_model_info().get("max_multimodal_batch_items")
            if raw_model_max is not None:
                model_max = max(1, int(raw_model_max))
        except (AttributeError, TypeError, ValueError):
            model_max = None
        batch_size = max(
            1,
            min(
                self.config.batch_size,
                provider_max,
                model_max if model_max is not None else provider_max,
                self._learned_multimodal_batch_size
                if self._learned_multimodal_batch_size is not None
                else provider_max,
            ),
        )
        if batch_size < self.config.batch_size:
            self.logger.info(
                "Clamped multimodal batch_size %d -> %d "
                "(binding=%s, model=%s, provider_max=%d, model_max=%s)",
                self.config.batch_size,
                batch_size,
                self.config.binding,
                self.config.model,
                provider_max,
                model_max,
            )
        all_embeddings: List[List[float]] = []
        start = 0
        completed_batches = 0
        expected_dim: int | None = None

        # A cursor loop lets a provider correct an outdated/unknown model cap.
        # Only the failed slice is retried; vectors from prior batches remain.
        while start < len(contents):
            batch = contents[start : start + batch_size]
            request = EmbeddingRequest(
                texts=[],
                model=self.config.model,
                dimensions=self.config.dim or None,
                contents=batch,
                enable_fusion=False,
            )
            try:
                response = await self.adapter.embed(request)
            except Exception as exc:
                explicit_limit = _batch_limit_from_error(exc)
                if not _is_adaptable_batch_error(exc) or len(batch) <= 1:
                    self.logger.error(
                        "Multimodal embedding batch failed "
                        "(binding=%s, model=%s, start_index=%d, batch_items=%d): %s",
                        self.config.binding,
                        self.config.model,
                        start,
                        len(batch),
                        exc,
                    )
                    raise

                if explicit_limit is not None and explicit_limit < len(batch):
                    smaller_batch_size = min(batch_size, explicit_limit)
                else:
                    smaller_batch_size = max(1, len(batch) // 2)
                if smaller_batch_size >= batch_size:
                    raise
                self.logger.warning(
                    "Multimodal embedding provider rejected batch_items=%d; "
                    "retrying the same slice with batch_size=%d "
                    "(binding=%s, model=%s): %s",
                    len(batch),
                    smaller_batch_size,
                    self.config.binding,
                    self.config.model,
                    exc,
                )
                batch_size = smaller_batch_size
                self._learned_multimodal_batch_size = smaller_batch_size
                continue

            total_batches = completed_batches + (
                (len(contents) - start + batch_size - 1) // batch_size
            )
            validated = validate_embedding_batch(
                response.embeddings,
                expected_count=len(batch),
                binding=self.config.binding,
                model=self.config.model,
                batch_index=completed_batches + 1,
                total_batches=total_batches,
                start_index=start,
            )
            batch_dim = len(validated[0]) if validated else 0
            if expected_dim is None:
                expected_dim = batch_dim
            elif batch_dim != expected_dim:
                raise ValueError(
                    "Embedding provider returned inconsistent vector dimensions "
                    f"across multimodal batches (binding={self.config.binding}, "
                    f"model={self.config.model}): expected {expected_dim}, got {batch_dim}."
                )
            all_embeddings.extend(validated)
            start += len(batch)
            completed_batches += 1

            if progress_callback:
                try:
                    progress_callback(completed_batches, total_batches)
                except Exception:
                    pass

            if start < len(contents) and self.config.batch_delay > 0:
                await asyncio.sleep(self.config.batch_delay)

        return all_embeddings

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.embed(texts))

        # Shared executor instead of a fresh ThreadPoolExecutor per call
        # (thread creation on every sync embedding was pure overhead).
        with _sync_embed_executor() as executor:
            future = executor.submit(asyncio.run, self.embed(texts))
            return future.result()


_client: Optional[EmbeddingClient] = None


def get_embedding_client(config: Optional[EmbeddingConfig] = None) -> EmbeddingClient:
    global _client
    resolved_config = config or get_embedding_config()
    if _client is None or _client.config != resolved_config:
        _client = EmbeddingClient(resolved_config)
    return _client


def reset_embedding_client() -> None:
    global _client
    _client = None
