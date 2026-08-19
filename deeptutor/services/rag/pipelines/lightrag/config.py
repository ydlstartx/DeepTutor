"""Bridge DeepTutor's runtime config into LightRAG / RAG-Anything.

LightRAG (HKUDS/LightRAG) is a text knowledge-graph RAG engine; its multimodal
story is RAG-Anything (HKUDS/RAG-Anything), built on top of LightRAG. The
``lightrag`` provider uses RAG-Anything so multimodal content (the parse layer's
``content_list``) becomes graph entities, while text-only documents fall back to
a plain text insert.

This module is the decoupling seam: it exposes availability + mode helpers and
builds the three adapters LightRAG needs from DeepTutor's already-resolved LLM /
embedding clients. It imports neither RAG-Anything nor LightRAG at module load —
the adapter builders import ``lightrag.utils`` lazily (only the embedding wrapper
needs it), and engine construction lives in ``engine.py``.

Decoupling notes:
* ``llm_model_func`` / ``vision_model_func`` wrap DeepTutor's unified model
  callables and DROP LightRAG's internal kwargs (``hashing_kv``,
  ``keyword_extraction``, …) so they never leak into ``factory.complete``.
* ``embedding_func`` reuses DeepTutor's embedding client, wrapped in LightRAG's
  ``EmbeddingFunc`` with the active model's dimension.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .worker import OwnerLoopBridge

logger = logging.getLogger(__name__)

# LightRAG's native retrieval modes. ``hybrid`` (KG + vector) is the safest
# general default and matches the shared per-KB ``search_mode`` default.
SUPPORTED_MODES = ("naive", "local", "global", "hybrid", "mix")
DEFAULT_MODE = "hybrid"

_ENTITY_EXTRACTION_MARKER = (
    "You are a Knowledge Graph Specialist responsible for extracting entities and relationships"
)
_TUPLE_DELIMITER = "<|#|>"
_COMPLETION_DELIMITER = "<|COMPLETE|>"
_FORMAT_RETRY_PROMPT = """The previous extraction output did not follow the required record format.
Re-output the complete extraction, correcting every malformed record.
Output only entity records with exactly 4 <|#|>-separated fields, relation records with exactly 5 fields, and finish with <|COMPLETE|>.
Do not add Markdown fences, explanations, or introductory text."""

# Conservative cap for the embedding wrapper when the model doesn't advertise one.
_DEFAULT_MAX_TOKEN_SIZE = 8192


class LightRagNotAvailableError(RuntimeError):
    """Raised when the optional ``raganything`` dependency is not installed."""


class LightRagNotConfiguredError(RuntimeError):
    """Raised when DeepTutor's LLM / embedding config can't back LightRAG."""


def is_lightrag_available() -> bool:
    """True when RAG-Anything (which bundles LightRAG) can be imported.

    Opt-in extra: ``pip install 'deeptutor[rag-lightrag]'``. Until installed the
    provider is hidden / blocked in the UI.
    """
    return importlib.util.find_spec("raganything") is not None


def normalize_mode(mode: str | None) -> str:
    """Coerce a stored ``search_mode`` to a valid LightRAG query mode.

    The per-KB ``search_mode`` field is shared across engines; anything that
    isn't a LightRAG mode falls back to :data:`DEFAULT_MODE`.
    """
    candidate = (mode or "").strip().lower()
    return candidate if candidate in SUPPORTED_MODES else DEFAULT_MODE


def query_kwargs_from_settings() -> dict:
    """Extra ``aquery`` kwargs (top_k, response_type) from runtime settings.

    Returned as a dict so the engine can pass them through to LightRAG's
    ``QueryParam`` and gracefully drop them if an older RAG-Anything rejects a
    kwarg. Empty on any read error.
    """
    try:
        from deeptutor.services.config import load_lightrag_settings

        settings = load_lightrag_settings()
        return {
            "top_k": int(settings.get("top_k", 60)),
            "response_type": str(settings.get("response_type") or "Multiple Paragraphs"),
        }
    except Exception:
        return {}


def indexing_kwargs_from_settings() -> dict:
    """LightRAG construction knobs that control indexing throughput/cost."""
    try:
        from deeptutor.services.config import load_lightrag_settings

        settings = load_lightrag_settings()
        return {
            "llm_model_max_async": int(settings.get("llm_concurrency", 8)),
            "embedding_func_max_async": int(settings.get("embedding_concurrency", 2)),
            "max_parallel_insert": int(settings.get("multimodal_concurrency", 8)),
            "entity_extract_max_gleaning": int(settings.get("entity_extract_max_gleaning", 0)),
            "chunk_token_size": int(settings.get("chunk_token_size", 1400)),
            "chunk_overlap_token_size": int(settings.get("chunk_overlap_token_size", 80)),
            "embedding_batch_num": int(settings.get("embedding_batch_num", 20)),
            "force_llm_summary_on_merge": int(settings.get("force_llm_summary_on_merge", 16)),
        }
    except Exception:
        return {
            "llm_model_max_async": 8,
            "embedding_func_max_async": 2,
            "max_parallel_insert": 8,
            "entity_extract_max_gleaning": 0,
            "chunk_token_size": 1400,
            "chunk_overlap_token_size": 80,
            "embedding_batch_num": 20,
            "force_llm_summary_on_merge": 16,
        }


def _entity_extraction_format_score(result: Any) -> tuple[bool, int]:
    """Return ``(is_parseable, valid_record_count)`` for extraction output.

    LightRAG 1.4.x drops entity records that do not have exactly four fields
    and relation records that do not have exactly five. Detect those failures
    before the result enters LightRAG's persistent cache so only malformed
    chunks pay for a corrective request.
    """
    if not isinstance(result, str):
        return False, 0

    has_completion = _COMPLETION_DELIMITER in result.upper()
    valid_records = 0
    malformed = False
    records = result.replace(_COMPLETION_DELIMITER, "\n").replace(
        _COMPLETION_DELIMITER.lower(), "\n"
    )
    for raw_line in records.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(_TUPLE_DELIMITER)]
        record_type = fields[0].lower() if fields else ""
        if record_type == "entity" and len(fields) == 4 and all(fields[1:]):
            valid_records += 1
        elif record_type in {"relation", "relationship"} and len(fields) == 5 and all(fields[1:]):
            valid_records += 1
        else:
            malformed = True

    return has_completion and not malformed, valid_records


def build_llm_model_func(*, io_bridge: OwnerLoopBridge | None = None):
    """Wrap DeepTutor's unified LLM callable for LightRAG.

    Drops LightRAG's internal kwargs while preserving explicit ``messages``.
    Entity-extraction responses are format-checked and only malformed chunks
    are retried, avoiding the cost of an unconditional gleaning pass.
    """
    from deeptutor.services.llm import get_llm_client

    base = get_llm_client().get_model_func()

    async def llm_model_func(
        prompt="",
        system_prompt=None,
        history_messages=None,
        messages=None,
        **_ignored,
    ):
        async def request(
            request_prompt: str,
            request_history: list[dict[str, Any]],
        ):
            return await base(
                request_prompt,
                system_prompt=system_prompt,
                history_messages=request_history,
                messages=messages,
            )

        original_prompt = prompt or ""
        original_history = list(history_messages or [])

        async def run_request(request_prompt: str, request_history: list[dict[str, Any]]):
            if io_bridge is not None:
                return await io_bridge.run(lambda: request(request_prompt, request_history))
            return await request(request_prompt, request_history)

        result = await run_request(original_prompt, original_history)
        is_extraction = (
            messages is None
            and isinstance(system_prompt, str)
            and _ENTITY_EXTRACTION_MARKER in system_prompt
        )
        if not is_extraction:
            return result

        parseable, valid_count = _entity_extraction_format_score(result)
        if parseable:
            return result

        logger.warning(
            "LightRAG entity extraction returned malformed output (%d valid records); "
            "retrying this chunk once",
            valid_count,
        )
        retry_history = [
            *original_history,
            {"role": "user", "content": original_prompt},
            {"role": "assistant", "content": str(result)},
        ]
        repaired = await run_request(_FORMAT_RETRY_PROMPT, retry_history)
        repaired_parseable, repaired_count = _entity_extraction_format_score(repaired)
        if (repaired_parseable and repaired_count >= valid_count) or (repaired_count > valid_count):
            return repaired
        return result

    return llm_model_func


def build_vision_model_func(*, io_bridge: OwnerLoopBridge | None = None):
    """Wrap DeepTutor's vision-capable callable for RAG-Anything's image step."""
    from deeptutor.services.llm import get_llm_client

    base = get_llm_client().get_vision_model_func()

    async def vision_model_func(
        prompt="",
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **_ignored,
    ):
        async def request():
            return await base(
                prompt or "",
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                image_data=image_data,
                messages=messages,
            )

        return await io_bridge.run(request) if io_bridge is not None else await request()

    return vision_model_func


def build_embedding_func(*, io_bridge: OwnerLoopBridge | None = None):
    """Wrap DeepTutor's embedding client in LightRAG's ``EmbeddingFunc``."""
    from lightrag.utils import EmbeddingFunc

    from deeptutor.services.embedding import get_embedding_client, get_embedding_config

    cfg = get_embedding_config()
    dim = int(getattr(cfg, "dim", 0) or 0)
    if not dim:
        raise LightRagNotConfiguredError(
            "No active embedding model with a known dimension. Configure one under "
            "Settings → Catalog before using a LightRAG knowledge base."
        )

    client = get_embedding_client()

    async def embedding_func(texts, context=None, **_ignored):
        import numpy as np

        # No context means no role, which is what the pinned LightRAG always
        # passes — defaulting to "document" would label queries as passages.
        input_type = {
            "query": "search_query",
            "document": "search_document",
        }.get(str(context or "").strip().lower())

        async def request():
            return await client.embed(texts, input_type=input_type)

        vectors = await io_bridge.run(request) if io_bridge is not None else await request()
        return np.asarray(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=int(getattr(cfg, "max_tokens", 0) or _DEFAULT_MAX_TOKEN_SIZE),
        func=embedding_func,
    )


__all__ = [
    "SUPPORTED_MODES",
    "DEFAULT_MODE",
    "LightRagNotAvailableError",
    "LightRagNotConfiguredError",
    "is_lightrag_available",
    "normalize_mode",
    "query_kwargs_from_settings",
    "indexing_kwargs_from_settings",
    "build_llm_model_func",
    "build_vision_model_func",
    "build_embedding_func",
]
