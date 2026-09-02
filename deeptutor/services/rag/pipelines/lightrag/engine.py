"""Thin adapters over the RAG-Anything / LightRAG Python APIs.

This is the ONLY module that imports ``raganything`` / ``lightrag``. Everything
version-sensitive lives here, so an API shift between releases is a one-file
fix. All imports are lazy so DeepTutor runs fine without the optional dependency
installed.

Full deployments build a RAG-Anything instance from DeepTutor's
LLM/vision/embedding adapters. Query-only deployments construct native LightRAG
directly, avoiding RAG-Anything's unused MinerU/PyTorch dependency tree while
opening the exact same per-KB stores and retaining every native query mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
import importlib.util
import inspect
import logging
import os
from pathlib import Path
from stat import S_IMODE
import tempfile
from typing import Any

from .config import (
    DEFAULT_MODE,
    build_embedding_func,
    build_llm_model_func,
    build_vision_model_func,
    indexing_kwargs_from_settings,
    lightrag_kwargs_from_settings,
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

# LightRAG wraps the injected LLM/embedding funcs in its own watchdog
# (priority_limit_async_func_call): a worker hard-cancels any call still
# running past 2× default_*_timeout, and a health check force-fails it 15s
# after that. The stock defaults (180s LLM / 30s embedding → 360s/60s kills)
# fire long before DeepTutor's own per-attempt wall-clock cap (900s,
# provider_core) and transient-retry policy can do their job — turning a
# recoverable slow or hung request into a hard document failure with no
# retry. Push the watchdog out so it stays a last-resort backstop:
# 2×900s ≈ two full DeepTutor attempts for the LLM, and 2×240s covers the
# embedding client's entire retry budget (6 attempts × 60s + backoff ≈ 390s).
_LIGHTRAG_LLM_TIMEOUT_S = 900
_LIGHTRAG_EMBEDDING_TIMEOUT_S = 240


def _is_xml_10_char(char: str) -> bool:
    codepoint = ord(char)
    return (
        codepoint in {0x09, 0x0A, 0x0D}
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _sanitize_xml_text(value: str) -> tuple[str, int]:
    """Repair characters that XML 1.0 cannot serialize.

    JSON decoding commonly turns an unescaped LaTeX ``\\beta`` / ``\\frac``
    into backspace/form-feed plus the remaining letters. Recover those two
    prefixes; replace every other illegal code point visibly instead of
    silently dropping source text.
    """
    replacements = {"\x08": r"\b", "\x0c": r"\f"}
    parts: list[str] = []
    changed = 0
    for char in value:
        if _is_xml_10_char(char):
            parts.append(char)
            continue
        parts.append(replacements.get(char, "\ufffd"))
        changed += 1
    return "".join(parts), changed


def _sanitize_xml_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return _sanitize_xml_text(value)
    if isinstance(value, list):
        cleaned: list[Any] = []
        changed = 0
        for item in value:
            next_item, item_changed = _sanitize_xml_value(item)
            cleaned.append(next_item)
            changed += item_changed
        return cleaned, changed
    if isinstance(value, tuple):
        cleaned_tuple: list[Any] = []
        changed = 0
        for item in value:
            next_item, item_changed = _sanitize_xml_value(item)
            cleaned_tuple.append(next_item)
            changed += item_changed
        return tuple(cleaned_tuple), changed
    if isinstance(value, dict):
        cleaned_dict: dict[Any, Any] = {}
        changed = 0
        for key, item in value.items():
            next_key, key_changed = _sanitize_xml_value(key)
            next_item, item_changed = _sanitize_xml_value(item)
            cleaned_dict[next_key] = next_item
            changed += key_changed + item_changed
        return cleaned_dict, changed
    return value, 0


def _sanitize_graph_for_xml(graph: Any) -> int:
    """Make graph ids and attributes XML-safe before GraphML persistence."""
    import networkx as nx

    changed = 0
    relabel: dict[Any, Any] = {}
    for node in graph.nodes:
        if not isinstance(node, str):
            continue
        clean_node, node_changed = _sanitize_xml_text(node)
        if not node_changed:
            continue
        if clean_node in graph and clean_node != node:
            raise ValueError(f"XML sanitization would merge graph nodes: {node!r}")
        relabel[node] = clean_node
        changed += node_changed
    if relabel:
        nx.relabel_nodes(graph, relabel, copy=False)

    attribute_maps = [graph.graph]
    attribute_maps.extend(data for _, data in graph.nodes(data=True))
    attribute_maps.extend(data for *_edge, data in graph.edges(data=True))
    for attributes in attribute_maps:
        for key, value in tuple(attributes.items()):
            clean_value, value_changed = _sanitize_xml_value(value)
            if value_changed:
                attributes[key] = clean_value
                changed += value_changed
    return changed


def _atomic_write_nx_graph(graph: Any, file_name: str, workspace: str = "_") -> None:
    """Write a verified GraphML file without exposing a partial target."""
    import networkx as nx

    target = Path(file_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_mode = S_IMODE(target.stat().st_mode) if target.exists() else None
    changed = _sanitize_graph_for_xml(graph)
    if changed:
        logger.warning(
            "[%s] Repaired %d XML-invalid character(s) before GraphML persistence",
            workspace,
            changed,
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        logger.info(
            "[%s] Writing graph with %d nodes, %d edges",
            workspace,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        nx.write_graphml(graph, temporary)
        # Import lazily to keep this adapter's optional-dependency surface
        # small while sharing the same structural validation used on reads.
        from .storage import validate_graphml_file

        validate_graphml_file(temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        if target_mode is not None:
            os.chmod(temporary, target_mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _install_atomic_networkx_storage() -> None:
    """Patch the pinned LightRAG NetworkX backend with safe persistence."""
    try:
        from lightrag.kg.networkx_impl import NetworkXStorage
    except (ImportError, ModuleNotFoundError):
        return

    if getattr(NetworkXStorage, "_deeptutor_atomic_graphml", False):
        return

    original_index_done = NetworkXStorage.index_done_callback

    async def strict_index_done(self: Any) -> bool:
        result = await original_index_done(self)
        if result is False:
            raise RuntimeError(
                f"LightRAG failed to persist GraphML safely: {self._graphml_xml_file}"
            )
        return result

    NetworkXStorage.write_nx_graph = staticmethod(_atomic_write_nx_graph)
    NetworkXStorage.index_done_callback = strict_index_done
    NetworkXStorage._deeptutor_atomic_graphml = True


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


def _accepts(target: Any, name: str) -> bool:
    """Whether ``target``'s constructor takes a keyword called *name*.

    The settings knobs below ride on RAG-Anything parameters that arrived in
    different releases — ``lightrag_kwargs`` only exists from ~1.2.5, while the
    supported range starts at 1.0.1. Asking first keeps an older install
    working on RAG-Anything's own defaults instead of dying with a TypeError
    that takes the whole LightRAG engine down. Same defensive posture the query
    path already takes for ``QueryParam`` kwargs.
    """
    import inspect

    try:
        return name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _drop_unsupported(
    target: Any,
    kwargs: dict[str, Any],
    *,
    what: str,
    package: str = "RAG-Anything",
) -> dict[str, Any]:
    supported = {key: value for key, value in kwargs.items() if _accepts(target, key)}
    for key in kwargs.keys() - supported.keys():
        logger.warning(
            "Installed %s does not accept %s=%r on %s; leaving it at "
            "the library default. Upgrade %s to use this setting.",
            package,
            key,
            kwargs[key],
            what,
            package,
        )
    return supported


def _build_config(config_cls: Any, working_dir: Path) -> Any:
    knobs = _drop_unsupported(config_cls, indexing_kwargs_from_settings(), what="RAGAnythingConfig")
    return config_cls(working_dir=str(working_dir), **knobs)


def _construct(
    rag_cls: Any,
    *,
    lightrag_overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    extra = dict(lightrag_overrides or {})
    if extra and _accepts(rag_cls, "lightrag_kwargs"):
        kwargs["lightrag_kwargs"] = extra
    elif extra:
        logger.warning(
            "Installed RAG-Anything has no lightrag_kwargs passthrough; %s stay "
            "at LightRAG's defaults. Upgrade raganything to use these settings.",
            ", ".join(sorted(extra)),
        )
    return rag_cls(**kwargs)


def _is_query_only() -> bool:
    from deeptutor.knowledge.policy import is_kb_query_only

    return is_kb_query_only()


class _NativeQueryRag:
    """RAG-Anything-shaped facade over native LightRAG retrieval.

    The pipeline intentionally consumes one small common interface for full and
    query-only images. This facade supplies only lifecycle and query methods —
    it has no insertion API, adding a provider-specific backstop behind the
    deployment-wide query-only policy.
    """

    def __init__(self, lightrag: Any) -> None:
        self.lightrag = lightrag
        self.working_dir = Path(lightrag.working_dir)
        self._ready = False

    async def _ensure_lightrag_initialized(self) -> dict[str, Any]:
        if self._ready:
            return {"success": True}
        try:
            await self.lightrag.initialize_storages()
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()
        except Exception as exc:
            return {"success": False, "error": f"Failed to initialize LightRAG: {exc}"}
        self._ready = True
        return {"success": True}

    async def aquery(self, question: str, mode: str | None = None, **kwargs: Any) -> Any:
        from lightrag import QueryParam

        query_kwargs = _drop_unsupported(
            QueryParam,
            kwargs,
            what="QueryParam",
            package="LightRAG",
        )
        param = QueryParam(mode=normalize_mode(mode) or DEFAULT_MODE, **query_kwargs)
        return await self.lightrag.aquery(question, param=param)

    async def finalize_storages(self) -> None:
        if not self._ready:
            return
        await self.lightrag.finalize_storages()
        self._ready = False


def _build_native_query_rag(
    working_dir: Path,
    *,
    lightrag_overrides: Mapping[str, Any],
    io_bridge: OwnerLoopBridge | None,
) -> _NativeQueryRag:
    from lightrag import LightRAG

    adapter_kwargs = {"io_bridge": io_bridge} if io_bridge is not None else {}
    params: dict[str, Any] = {
        "working_dir": str(working_dir),
        "llm_model_func": build_llm_model_func(**adapter_kwargs),
        "embedding_func": build_embedding_func(**adapter_kwargs),
        **lightrag_overrides,
    }
    params = _drop_unsupported(
        LightRAG,
        params,
        what="LightRAG",
        package="LightRAG",
    )
    return _NativeQueryRag(LightRAG(**params))


def build_rag(
    working_dir: Path,
    vector_storage: str | None = None,
    *,
    io_bridge: OwnerLoopBridge | None = None,
) -> Any:
    """Construct the deployment's LightRAG facade rooted at ``working_dir``.

    Full deployments use RAG-Anything for multimodal insertion. Query-only
    deployments use native LightRAG and therefore need neither RAG-Anything nor
    MinerU/PyTorch at runtime.

    ``vector_storage`` picks the vector-store engine ("nano" | "faiss"). The
    caller resolves it from the on-disk version's meta.json (falling back to
    "nano" for versions that predate the field), so an existing KB always
    opens with the engine it was built with regardless of the global default.
    ``io_bridge`` routes DeepTutor-owned network calls back to the service
    event loop when the instance runs inside an indexing worker thread.
    """
    # LightRAG's NetworkX backend writes GraphML directly to the live target
    # and swallows write failures. Install DeepTutor's guarded writer before
    # RAG-Anything constructs or loads any storage objects.
    _install_atomic_networkx_storage()

    engine_id = (vector_storage or DEFAULT_VECTOR_STORAGE).strip().lower()
    if engine_id not in VECTOR_STORAGE_CLASSES:
        logger.warning(
            "Unknown vector_storage %r; falling back to %r", engine_id, DEFAULT_VECTOR_STORAGE
        )
        engine_id = DEFAULT_VECTOR_STORAGE
    storage_cls = VECTOR_STORAGE_CLASSES[engine_id]

    lightrag_overrides = lightrag_kwargs_from_settings()
    # Keep LightRAG's call watchdog strictly a backstop behind DeepTutor's own
    # attempt cap + retries (see the constants above).
    lightrag_overrides["default_llm_timeout"] = _LIGHTRAG_LLM_TIMEOUT_S
    lightrag_overrides["default_embedding_timeout"] = _LIGHTRAG_EMBEDDING_TIMEOUT_S
    query_only = _is_query_only()
    if query_only:
        # LightRAG enables a persistent LLM response cache by default. A query
        # miss would therefore write into the published index tree even though
        # no indexing endpoint ran. Query-only deployments must keep retrieval
        # fully read-only, so skip that cache while retaining query embeddings
        # and every native retrieval mode.
        lightrag_overrides["enable_llm_cache"] = False
        lightrag_overrides["enable_llm_cache_for_entity_extract"] = False
    if storage_cls is not None:
        if not importlib.util.find_spec("faiss"):
            raise RuntimeError(
                "The 'faiss' vector storage engine requires the faiss-cpu package. "
                "Install it with `pip install faiss-cpu` (or reinstall with the "
                "rag-lightrag extra), or switch the engine back to 'nano'."
            )
        _install_lean_faiss_storage()
        # RAG-Anything forwards lightrag_kwargs into LightRAG(**params).
        lightrag_overrides["vector_storage"] = storage_cls

    if query_only:
        return _build_native_query_rag(
            working_dir,
            lightrag_overrides=lightrag_overrides,
            io_bridge=io_bridge,
        )

    from raganything import RAGAnything, RAGAnythingConfig

    config = _build_config(RAGAnythingConfig, working_dir)
    adapter_kwargs = {"io_bridge": io_bridge} if io_bridge is not None else {}
    funcs = {
        "llm_model_func": build_llm_model_func(**adapter_kwargs),
        "vision_model_func": build_vision_model_func(**adapter_kwargs),
        "embedding_func": build_embedding_func(**adapter_kwargs),
    }
    rag = _construct(
        RAGAnything,
        config=config,
        lightrag_overrides=lightrag_overrides,
        **funcs,
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
    cleaned_content, changed = _sanitize_xml_value(content_list)
    if changed:
        logger.warning(
            "LightRAG: repaired %d XML-invalid character(s) in parsed content for %s",
            changed,
            file_name,
        )
    await rag.insert_content_list(
        content_list=cleaned_content,
        file_path=file_name,
        doc_id=doc_id,
    )


def _managed_queue_funcs(lightrag: Any) -> Iterable[Callable[..., Any]]:
    """Yield each current LightRAG queue wrapper exactly once."""
    role_funcs = getattr(lightrag, "role_llm_funcs", {})
    candidates: list[object] = []
    if isinstance(role_funcs, Mapping):
        candidates.extend(role_funcs.values())

    embedding = getattr(lightrag, "embedding_func", None)
    candidates.append(getattr(embedding, "func", None))
    candidates.append(getattr(lightrag, "rerank_model_func", None))

    seen: set[int] = set()
    for candidate in candidates:
        if not callable(candidate) or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate


async def _shutdown_queues(lightrag: Any, *, cancel_pending: bool) -> None:
    """Bound cleanup of LightRAG's role, embedding, and rerank queues."""
    shutdowns: list[Awaitable[Any]] = []
    for func in _managed_queue_funcs(lightrag):
        shutdown = getattr(func, "shutdown", None)
        if callable(shutdown):
            # LightRAG 1.4.x exposes ``shutdown()`` with no arguments, while
            # newer queue wrappers accept graceful/timeout controls. Keep the
            # cleanup path compatible with the full supported 1.4.x range.
            shutdown_kwargs: dict[str, Any] = {}
            if _accepts(shutdown, "graceful"):
                shutdown_kwargs["graceful"] = not cancel_pending
            if _accepts(shutdown, "timeout"):
                shutdown_kwargs["timeout"] = 5.0
            result = shutdown(**shutdown_kwargs)
            if inspect.isawaitable(result):
                shutdowns.append(result)
    if not shutdowns:
        return

    results = await asyncio.gather(*shutdowns, return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(
            f"Failed to shut down {len(failures)} LightRAG managed queue(s)"
        ) from failures[0]


async def finalize(rag: Any, *, cancel_pending: bool) -> None:
    """Stop managed work before finalizing the facade's storage resources."""
    lightrag = getattr(rag, "lightrag", None)
    if lightrag is not None:
        await _shutdown_queues(lightrag, cancel_pending=cancel_pending)

    finalizer = getattr(rag, "finalize_storages", None)
    if not callable(finalizer):
        return
    result = finalizer()
    if inspect.isawaitable(result):
        await result


async def ensure_ready(rag: Any) -> None:
    """Ensure the full or query-only facade has initialized LightRAG stores."""
    if getattr(rag, "lightrag", None) is not None and not isinstance(rag, _NativeQueryRag):
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
    older RAG-Anything or LightRAG release that rejects one of these kwargs
    falls back to a mode-only query rather than failing the search.
    """
    resolved = normalize_mode(mode) or DEFAULT_MODE
    extra = query_kwargs_from_settings()
    await ensure_ready(rag)
    try:
        result = await rag.aquery(question, mode=resolved, **extra)
    except TypeError:
        if extra:
            logger.debug("LightRAG facade rejected extra query kwargs; retrying mode-only.")
            result = await rag.aquery(question, mode=resolved)
        else:
            raise
    return result if isinstance(result, str) else str(result)


async def query_with_sources(
    rag: Any, question: str, mode: str | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Return a LightRAG answer together with its structured provenance.

    Native query-only deployments can use LightRAG's combined ``aquery_llm``
    API, which returns the answer and retrieval records from one search. Full
    RAG-Anything deployments keep their facade-owned answer path (including VLM
    enhancement) and fetch provenance separately for compatibility.
    """
    if isinstance(rag, _NativeQueryRag):
        await ensure_ready(rag)
        complete_query = getattr(rag.lightrag, "aquery_llm", None)
        if callable(complete_query):
            from lightrag import QueryParam

            resolved = normalize_mode(mode) or DEFAULT_MODE
            extra = query_kwargs_from_settings()
            query_kwargs = _drop_unsupported(
                QueryParam,
                extra,
                what="QueryParam",
                package="LightRAG",
            )
            result = await complete_query(
                question,
                param=QueryParam(mode=resolved, **query_kwargs),
            )
            if isinstance(result, dict):
                llm_response = result.get("llm_response")
                if isinstance(llm_response, dict) and not llm_response.get("is_streaming"):
                    answer = llm_response.get("content")
                    return (answer if isinstance(answer, str) else ""), _query_data_to_sources(
                        result
                    )

    answer = await query(rag, question, mode)
    return answer, await query_sources(rag, question, mode)


async def query_sources(rag: Any, question: str, mode: str | None = None) -> list[dict[str, Any]]:
    """Fetch and normalize the structured records used by a LightRAG query.

    ``aquery_data`` was added by LightRAG after some supported RAG-Anything
    releases. Missing or failed provenance must not turn a successful answer
    into a failed user request, so older installations gracefully return no
    citations.
    """
    await ensure_ready(rag)
    lightrag = getattr(rag, "lightrag", None)
    aquery_data = getattr(lightrag, "aquery_data", None)
    if not callable(aquery_data):
        logger.debug("Installed LightRAG has no structured query data API.")
        return []

    resolved = normalize_mode(mode) or DEFAULT_MODE
    extra = query_kwargs_from_settings()
    try:
        from lightrag import QueryParam

        try:
            result = await aquery_data(question, param=QueryParam(mode=resolved, **extra))
        except TypeError:
            if not extra:
                raise
            logger.debug("LightRAG rejected extra provenance query kwargs; retrying mode-only.")
            result = await aquery_data(question, param=QueryParam(mode=resolved))
    except Exception as exc:
        logger.warning("LightRAG provenance lookup failed; omitting citations: %s", exc)
        return []

    return _query_data_to_sources(result)


def _query_data_to_sources(result: Any) -> list[dict[str, Any]]:
    """Map LightRAG's structured retrieval result to DeepTutor citations."""
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if not isinstance(data, dict):
        return []

    reference_paths = {
        str(record.get("reference_id")): str(record.get("file_path"))
        for record in data.get("references", [])
        if isinstance(record, dict) and record.get("reference_id") and record.get("file_path")
    }
    sources: list[dict[str, Any]] = []

    def source_path(record: dict[str, Any]) -> str:
        return str(
            record.get("file_path") or reference_paths.get(str(record.get("reference_id")), "")
        )

    def source_title(path: str, fallback: str) -> str:
        return Path(path).name if path else fallback

    for record in data.get("chunks", []):
        if not isinstance(record, dict):
            continue
        path = source_path(record)
        content = str(record.get("content") or "")
        chunk_id = str(record.get("chunk_id") or "")
        if not (path or content or chunk_id):
            continue
        sources.append(
            {
                "title": source_title(path, "LightRAG chunk"),
                "content": content[:200],
                "source": path,
                "page": str(record.get("page") or ""),
                "chunk_id": chunk_id,
                "reference_id": str(record.get("reference_id") or ""),
            }
        )

    for record in data.get("entities", []):
        if not isinstance(record, dict):
            continue
        path = source_path(record)
        entity_id = str(record.get("entity_name") or record.get("entity_id") or "")
        description = str(record.get("description") or "")
        if not (path or entity_id or description):
            continue
        sources.append(
            {
                "title": entity_id or source_title(path, "LightRAG entity"),
                "content": description[:200],
                "source": path,
                "page": str(record.get("page") or ""),
                "entity_id": entity_id,
                "entity_type": str(record.get("entity_type") or ""),
                "source_id": str(record.get("source_id") or ""),
                "reference_id": str(record.get("reference_id") or ""),
            }
        )

    for record in data.get("relationships", []):
        if not isinstance(record, dict):
            continue
        path = source_path(record)
        source_id = str(record.get("src_id") or "")
        target_id = str(record.get("tgt_id") or "")
        relation_id = str(record.get("relation_id") or f"{source_id}->{target_id}")
        description = str(record.get("description") or "")
        if not (path or relation_id or description):
            continue
        sources.append(
            {
                "title": relation_id or source_title(path, "LightRAG relationship"),
                "content": description[:200],
                "source": path,
                "page": str(record.get("page") or ""),
                "relation_id": relation_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "source_id": str(record.get("source_id") or ""),
                "reference_id": str(record.get("reference_id") or ""),
            }
        )

    return sources


__all__ = [
    "build_rag",
    "ensure_ready",
    "finalize",
    "insert",
    "query",
    "query_with_sources",
]
