"""Thin adapter over the RAG-Anything / LightRAG Python API.

This is the ONLY module that imports ``raganything`` / ``lightrag``. Everything
version-sensitive lives here, so an API shift between releases is a one-file
fix. All imports are lazy so DeepTutor runs fine without the optional dependency
installed.

A RAG-Anything instance is built from DeepTutor's LLM/vision/embedding adapters
(see ``config.py``) over a per-KB ``working_dir``. Documents are inserted as a
MinerU-style ``content_list`` (produced upstream by the parse layer), so the
multimodal step never re-parses anything; retrieval delegates to LightRAG's
native query modes.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MODE,
    build_embedding_func,
    build_llm_model_func,
    build_vision_model_func,
    normalize_mode,
    query_kwargs_from_settings,
)
from .worker import OwnerLoopBridge

logger = logging.getLogger(__name__)

# deeptutor-level vector-storage ids -> LightRAG's registered class names.
# "nano" maps to None: NanoVectorDBStorage is LightRAG's own default, so no
# kwarg is passed (least surface for RAG-Anything version drift).
VECTOR_STORAGE_CLASSES = {
    "nano": None,
    "faiss": "FaissVectorDBStorage",
}
DEFAULT_VECTOR_STORAGE = "nano"


def _install_lean_faiss_storage() -> None:
    """Swap LightRAG's FaissVectorDBStorage for a RAM-lean subclass.

    Upstream ``_load_faiss_index`` reifies every stored vector as a Python
    float list (``meta["__vector__"]``) — ~80 KB per 2560-dim record held for
    the process lifetime, i.e. several GB for a KB with tens of thousands of
    relationships; ``upsert`` does the same for fresh inserts. That reified
    copy is redundant: the Faiss index already holds the vectors. The two
    ``__vector__`` consumers are covered — ``_remove_faiss_ids`` upstream
    falls back to ``index.reconstruct()``, and ``get_vectors_by_ids`` is
    overridden here to reconstruct on demand.

    The subclass keeps the upstream class name so LightRAG's registry lookup
    (``STORAGES`` + lazy import) and its logs are unaffected. lightrag-hku is
    pinned (``<1.5``), so the copied load logic cannot drift silently.
    """
    import json as _json
    import os as _os

    import faiss
    from lightrag.kg import faiss_impl
    from lightrag.utils import logger as _lr_logger

    upstream = faiss_impl.FaissVectorDBStorage
    if getattr(upstream, "_deeptutor_lean", False):
        return

    class LeanFaissVectorDBStorage(upstream):
        _deeptutor_lean = True

        def _load_faiss_index(self):
            # Identical to upstream except: no per-row reconstruct() into
            # "__vector__" — that line is what inflates RAM by ~80 KB/record.
            if not _os.path.exists(self._faiss_index_file):
                _lr_logger.warning(
                    f"[{self.workspace}] No existing Faiss index file found for {self.namespace}"
                )
                return

            dim_mismatch = False
            try:
                self._index = faiss.read_index(self._faiss_index_file)

                if self._index.d != self._dim:
                    error_msg = (
                        f"Dimension mismatch: loaded Faiss index has dimension {self._index.d}, "
                        f"but embedding function expects dimension {self._dim}. "
                        f"Please ensure the embedding model matches the stored index or rebuild the index."
                    )
                    _lr_logger.error(error_msg)
                    dim_mismatch = True
                    raise ValueError(error_msg)

                with open(self._meta_file, "r", encoding="utf-8") as f:
                    stored_dict = _json.load(f)

                self._id_to_meta = {}
                for fid_str, meta in stored_dict.items():
                    fid = int(fid_str)
                    if fid >= self._index.ntotal:
                        _lr_logger.warning(
                            f"[{self.workspace}] Skipping metadata row fid={fid}: "
                            f"exceeds index size ({self._index.ntotal})"
                        )
                        continue
                    self._id_to_meta[fid] = meta

                _lr_logger.info(
                    f"[{self.workspace}] Faiss index loaded with {self._index.ntotal} vectors from {self._faiss_index_file}"
                )
            except Exception as e:
                if dim_mismatch:
                    raise
                _lr_logger.error(f"[{self.workspace}] Failed to load Faiss index or metadata: {e}")
                _lr_logger.warning(f"[{self.workspace}] Starting with an empty Faiss index.")
                self._index = faiss.IndexFlatIP(self._dim)
                self._id_to_meta = {}

        async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
            await super().upsert(data)
            # Drop the per-record embedding lists upstream keeps in RAM; the
            # Faiss index already holds them and the on-disk meta never did.
            for meta in self._id_to_meta.values():
                meta.pop("__vector__", None)

        async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
            # Upstream only serves "__vector__" from RAM; reconstruct on
            # demand instead (bounded to the ids actually asked for).
            if not ids:
                return {}
            vectors: dict[str, list[float]] = {}
            for custom_id in ids:
                fid = self._find_faiss_id_by_custom_id(custom_id)
                if fid is None or fid not in self._id_to_meta:
                    continue
                meta = self._id_to_meta[fid]
                vec = meta.get("__vector__")
                if vec is None and fid < self._index.ntotal:
                    vec = self._index.reconstruct(fid).tolist()
                if vec is not None:
                    vectors[custom_id] = vec
            return vectors

    LeanFaissVectorDBStorage.__name__ = "FaissVectorDBStorage"
    LeanFaissVectorDBStorage.__qualname__ = "FaissVectorDBStorage"
    faiss_impl.FaissVectorDBStorage = LeanFaissVectorDBStorage


def build_rag(
    working_dir: Path,
    vector_storage: str | None = None,
    *,
    io_bridge: OwnerLoopBridge | None = None,
) -> Any:
    """Construct a RAG-Anything instance rooted at ``working_dir``.

    Pinned to RAG-Anything's config-based constructor; this is the single spot
    to touch if its API changes between releases.

    ``vector_storage`` picks the vector-store engine ("nano" | "faiss"). The
    caller resolves it from the on-disk version's meta.json (falling back to
    "nano" for versions that predate the field), so an existing KB always
    opens with the engine it was built with regardless of the global default.
    ``io_bridge`` routes DeepTutor-owned network calls back to the service
    event loop when the instance runs inside an indexing worker thread.
    """
    from raganything import RAGAnything, RAGAnythingConfig

    engine_id = (vector_storage or DEFAULT_VECTOR_STORAGE).strip().lower()
    if engine_id not in VECTOR_STORAGE_CLASSES:
        logger.warning(
            "Unknown vector_storage %r; falling back to %r", engine_id, DEFAULT_VECTOR_STORAGE
        )
        engine_id = DEFAULT_VECTOR_STORAGE
    storage_cls = VECTOR_STORAGE_CLASSES[engine_id]

    lightrag_kwargs: dict[str, Any] = {}
    if storage_cls is not None:
        if not importlib.util.find_spec("faiss"):
            raise RuntimeError(
                "The 'faiss' vector storage engine requires the faiss-cpu package. "
                "Install it with `pip install faiss-cpu` (or reinstall with the "
                "rag-lightrag extra), or switch the engine back to 'nano'."
            )
        _install_lean_faiss_storage()
        # RAG-Anything forwards lightrag_kwargs into LightRAG(**params).
        lightrag_kwargs["vector_storage"] = storage_cls

    config = RAGAnythingConfig(working_dir=str(working_dir))
    adapter_kwargs = {"io_bridge": io_bridge} if io_bridge is not None else {}
    rag = RAGAnything(
        config=config,
        llm_model_func=build_llm_model_func(**adapter_kwargs),
        vision_model_func=build_vision_model_func(**adapter_kwargs),
        embedding_func=build_embedding_func(**adapter_kwargs),
        lightrag_kwargs=lightrag_kwargs,
    )
    # DeepTutor always feeds RAG-Anything a pre-parsed ``content_list`` (the
    # parse layer runs upstream via DeepTutor's own ParseService), so
    # RAG-Anything's bundled document parser is never invoked. Its LightRAG init
    # nevertheless runs a one-time installation check on its *default* parser
    # (``mineru``); when MinerU isn't installed that check hard-fails indexing
    # with "Parser 'mineru' is not properly installed" — even though the user
    # picked an entirely different parse engine (see issue #594). Marking the
    # check as already satisfied skips that spurious gate for a parser we don't
    # use, while leaving the real pre-parsed insert path untouched.
    rag._parser_installation_checked = True
    return rag


async def insert(rag: Any, content_list: list[dict], *, file_name: str, doc_id: str) -> None:
    """Insert a pre-parsed ``content_list`` (multimodal-aware, no re-parsing)."""
    await rag.insert_content_list(
        content_list=content_list,
        file_path=file_name,
        doc_id=doc_id,
    )


async def ensure_ready(rag: Any) -> None:
    """Ensure RAG-Anything has an initialized LightRAG instance."""
    if getattr(rag, "lightrag", None) is not None:
        return

    initializer = getattr(rag, "_ensure_lightrag_initialized", None)
    if initializer is None:
        return

    result = await initializer()
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(result.get("error") or "Failed to initialize LightRAG")


async def query(rag: Any, question: str, mode: str | None = None) -> str:
    """Run a LightRAG query and return the synthesized answer string.

    Extra knobs (top_k, response_type) from the lightrag.json slice ride into
    LightRAG's ``QueryParam`` via aquery's ``**kwargs``. Wiring is defensive: an
    older RAG-Anything that rejects one of these kwargs falls back to a
    mode-only query rather than failing the search.
    """
    resolved = normalize_mode(mode) or DEFAULT_MODE
    extra = query_kwargs_from_settings()
    await ensure_ready(rag)
    try:
        result = await rag.aquery(question, mode=resolved, **extra)
    except TypeError:
        if extra:
            logger.debug("RAG-Anything rejected extra query kwargs; retrying mode-only.")
            result = await rag.aquery(question, mode=resolved)
        else:
            raise
    return result if isinstance(result, str) else str(result)


__all__ = ["build_rag", "insert", "ensure_ready", "query"]
