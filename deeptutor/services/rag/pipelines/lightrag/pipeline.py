"""LightRAG-backed RAG pipeline orchestration.

Implements the same contract as :class:`LlamaIndexPipeline` (see
``..base.RAGPipeline``) but delegates indexing/retrieval to RAG-Anything /
LightRAG. Each KB owns a self-contained LightRAG store under its ``version-N``
directory (see ``storage``).

Documents are turned into a MinerU-style ``content_list`` by DeepTutor's shared
parse layer (``deeptutor/services/parsing``) — the same bridge the question
extractor uses — so multimodal parsing stays a decoupled, cached, engine-
pluggable concern and this pipeline only ever feeds LightRAG ready content.

LightRAG is an optional dependency: every method fails with a clear, actionable
message when it is not installed instead of an opaque ``ImportError``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from pathlib import Path
import shutil
import traceback
from typing import Any, Dict, List, Optional

from deeptutor.runtime.home import get_runtime_data_root
from deeptutor.services.rag.index_versioning import (
    resolve_storage_dir_for_read,
    resolve_storage_dir_for_rebuild,
)
from deeptutor.services.rag.kb_paths import resolve_kb_dir

from . import block_policy, engine, storage
from . import config as lr_config
from .worker import OwnerLoopBridge, run_in_shared_worker_loop

logger = logging.getLogger(__name__)

DEFAULT_KB_BASE_DIR = str(get_runtime_data_root() / "knowledge_bases")


class LightRagPipeline:
    """Index/retrieve KB content via RAG-Anything / LightRAG."""

    def __init__(self, kb_base_dir: Optional[str] = None, **_: Any) -> None:
        self.logger = logging.getLogger(__name__)
        self.kb_base_dir = kb_base_dir or DEFAULT_KB_BASE_DIR
        # One RAG-Anything instance per on-disk version. Building one reloads
        # every LightRAG store (graph + vector index + KV JSONs); doing that on
        # EVERY query blocked the event loop for minutes on large KBs and, with
        # nano-vectordb, spiked memory until the OS killed the process.
        # LightRAG storages are built for concurrent in-process use and reload
        # on cross-process update flags, so reuse is safe.
        # All entries are created and consumed on the shared worker loop (see
        # ``worker.run_in_shared_worker_loop``), which serializes access — no
        # lock is needed here — and keeps LightRAG's process-global asyncio
        # locks bound to a loop that outlives any single request.
        self._rag_cache: dict[tuple[str, str, str], tuple[Any, asyncio.AbstractEventLoop]] = {}

    # ----- helpers --------------------------------------------------------

    async def _get_rag(
        self,
        root_dir: Path,
        *,
        io_bridge: OwnerLoopBridge,
        owner_loop: asyncio.AbstractEventLoop,
    ) -> Any:
        """Return the cached RAG instance for ``root_dir``, building it once.

        Must run on the shared worker loop. Cache key: version dir +
        vector-storage engine + active embedding identity — switching the
        embedding model mid-process must not reuse an instance that still
        embeds with the old model. Entries are additionally pinned to the
        owner loop their network bridge targets: a cached bridge pointing at
        a dead owner loop (e.g. a previous ``asyncio.run``) is useless, so
        such an entry is rebuilt.
        """
        from deeptutor.services.rag.embedding_signature import (
            signature_from_embedding_config,
        )

        engine_id = storage.read_vector_storage(root_dir)
        signature = signature_from_embedding_config()
        sig_key = f"{getattr(signature, 'model', '')}:{getattr(signature, 'dim', '')}"
        key = (str(Path(root_dir).resolve()), engine_id, sig_key)
        cached = self._rag_cache.get(key)
        if cached is not None and cached[1] is owner_loop:
            rag = cached[0]
        else:
            rag = engine.build_rag(storage.working_dir(root_dir), engine_id, io_bridge=io_bridge)
            self._rag_cache[key] = (rag, owner_loop)
        # Ready before use: RAG-Anything's init is not concurrency-guarded;
        # the shared loop serializes this against other first uses.
        await engine.ensure_ready(rag)
        return rag

    async def _drop_rag(self, root_dir: Path) -> None:
        """Evict cached instances for ``root_dir`` (failed/partial builds)."""

        async def job(_bridge: OwnerLoopBridge) -> None:
            prefix = str(Path(root_dir).resolve())
            doomed = [key for key in self._rag_cache if key[0] == prefix]
            for key in doomed:
                self._rag_cache.pop(key, None)

        await run_in_shared_worker_loop(job)

    def _ensure_available(self) -> None:
        if not lr_config.is_lightrag_available():
            from deeptutor.knowledge.policy import is_kb_query_only

            extra = "rag-lightrag-query" if is_kb_query_only() else "rag-lightrag"
            raise lr_config.LightRagNotAvailableError(
                "LightRAG is not installed. Install it with "
                f"`pip install 'deeptutor[{extra}]'` to use LightRAG knowledge bases."
            )

    def _resolve_mode(self, kb_name: str, kwargs: dict[str, Any]) -> str:
        from ..modes import resolve_kb_mode

        return resolve_kb_mode(
            self.kb_base_dir,
            kb_name,
            storage.PROVIDER,
            explicit=kwargs.get("mode"),
            supported=lr_config.SUPPORTED_MODES,
            default=lr_config.DEFAULT_MODE,
        )

    def _cleanup_failed_version_dir(self, root_dir: Path) -> None:
        try:
            if root_dir.is_dir() and not (root_dir / storage.META_FILENAME).exists():
                shutil.rmtree(root_dir)
        except Exception as exc:  # pragma: no cover - best-effort
            self.logger.warning("Could not clean up failed version dir %s: %s", root_dir, exc)

    @staticmethod
    def _raise_if_corrupt_index(root_dir: Path) -> None:
        problem = storage.graph_integrity_error(root_dir)
        if problem:
            raise RuntimeError(
                "LightRAG index is corrupted and cannot be updated safely: "
                f"{problem}. Rebuild the knowledge base before adding documents."
            )

    async def _ingest(
        self,
        rag: Any,
        file_paths: List[str],
        *,
        io_bridge: OwnerLoopBridge,
        progress_callback: Callable[[int, int], Any] | None = None,
    ) -> int:
        """Parse files concurrently, then insert them into LightRAG in order.

        Returns the number of documents successfully inserted. Per-file failures
        are logged and skipped so one bad document doesn't abort the batch.
        Parsing is bounded by ``max_concurrent_files`` and overlaps the serial
        graph writes. Inserts deliberately remain serial: LightRAG's JSON/graph
        stores are mutable shared state and cannot safely be written by several
        document tasks at once.
        """
        from deeptutor.services.parsing import ParserError, get_parse_service

        parse_service = get_parse_service()
        indexing_settings = lr_config.indexing_kwargs_from_settings()
        try:
            max_concurrent_files = max(
                1,
                int(indexing_settings.get("max_concurrent_files", 1)),
            )
        except (TypeError, ValueError):
            max_concurrent_files = 1
        parse_slots = asyncio.Semaphore(max_concurrent_files)

        async def parse_file(path: Path) -> tuple[Any | None, ParserError | None]:
            async with parse_slots:
                io_bridge.raise_if_cancelled()
                try:
                    document = await asyncio.to_thread(parse_service.parse, path)
                except ParserError as exc:
                    return None, exc
                io_bridge.raise_if_cancelled()
                return document, None

        # Create all tasks up front so later documents can be parsed while the
        # worker loop serially inserts an earlier one. Awaiting in input order
        # keeps progress and graph contents deterministic.
        parse_tasks = [asyncio.create_task(parse_file(Path(path))) for path in file_paths]
        inserted = 0
        total = len(file_paths)
        try:
            for path, task in zip(map(Path, file_paths), parse_tasks):
                io_bridge.raise_if_cancelled()
                doc, parse_error = await task
                if parse_error is not None:
                    self.logger.warning(
                        "LightRAG: parse failed for %s: %s",
                        path.name,
                        parse_error,
                    )
                    continue

                accepted_ledger: dict[str, Any] | None = None
                attempt_id: str | None = None
                if doc.blocks:
                    document_id = doc.source_hash or path.stem
                    decision = block_policy.prepare_content_list(
                        doc.blocks,
                        engine=doc.engine,
                        source_hash=doc.source_hash,
                        parser_signature=doc.parser_signature,
                    )
                    if decision.ledger is not None:
                        outcome = "unknown_types" if decision.unknown_type_counts else "accepted"
                        _, attempt_id = block_policy.write_attempt_ledger(
                            Path(rag.working_dir).parent,
                            document_id,
                            decision.ledger,
                            outcome=outcome,
                        )
                        counts = decision.ledger["counts"]
                        self.logger.info(
                            "MinerU block policy %s raw=%d filtered=%d eligible=%d unknown=%d",
                            block_policy.POLICY_ID,
                            counts["raw_total"],
                            counts["filtered_total"],
                            counts["eligible_total"],
                            counts["unknown_total"],
                        )
                        if decision.unknown_type_counts:
                            self.logger.warning(
                                "LightRAG: %s has MinerU block types with no policy "
                                "entry (%s); indexing them unclassified",
                                path.name,
                                decision.unknown_summary(),
                            )
                    accepted_ledger = decision.ledger
                    content_list = decision.content_list
                else:
                    content_list = (
                        [{"type": "text", "text": doc.markdown, "page_idx": 0}]
                        if doc.markdown
                        else []
                    )

                if not content_list:
                    self.logger.warning("LightRAG: empty document skipped: %s", path.name)
                    continue

                await engine.insert(
                    rag,
                    content_list,
                    file_name=path.name,
                    doc_id=doc.source_hash or path.stem,
                )
                io_bridge.raise_if_cancelled()
                doc_error = storage.document_error(
                    Path(rag.working_dir), doc.source_hash or path.stem
                )
                if doc_error:
                    raise RuntimeError(f"{path.name}: {doc_error}")
                if accepted_ledger is not None:
                    block_policy.write_decision_ledger(
                        Path(rag.working_dir),
                        doc.source_hash or path.stem,
                        accepted_ledger,
                        attempt_id=attempt_id,
                    )
                inserted += 1
                self.logger.info("LightRAG: inserted %s", path.name)
                if progress_callback is not None:
                    await io_bridge.call(progress_callback, inserted, total)
            return inserted
        finally:
            for task in parse_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*parse_tasks, return_exceptions=True)

    async def _run_indexing(
        self,
        working_dir: Path,
        file_paths: List[str],
        progress_callback: Callable[[int, int], Any] | None,
    ) -> int:
        """Run the complete local LightRAG indexing phase off the service loop.

        The RAG instance is constructed and consumed in one worker thread, so
        its mutable stores and asyncio primitives never cross event loops.
        DeepTutor-owned network calls and progress callbacks cross back through
        ``OwnerLoopBridge`` and remain responsive while local JSON storage is
        busy in the worker.

        The vector-storage engine is resolved up front (version meta.json pin,
        else the global setting) so the worker-built instance matches what
        later queries through ``_get_rag`` will reopen.
        """

        if storage.has_output(working_dir):
            self._raise_if_corrupt_index(working_dir)
        engine_id = storage.read_vector_storage(working_dir)

        async def job(io_bridge: OwnerLoopBridge) -> int:
            io_bridge.raise_if_cancelled()
            rag = engine.build_rag(working_dir, engine_id, io_bridge=io_bridge)
            failed = True
            try:
                result = await self._ingest(
                    rag,
                    file_paths,
                    io_bridge=io_bridge,
                    progress_callback=progress_callback,
                )
                failed = False
                return result
            finally:
                try:
                    await engine.finalize(rag, cancel_pending=failed)
                except BaseException:
                    if not failed:
                        raise
                    self.logger.exception(
                        "LightRAG resource cleanup failed while indexing was aborting"
                    )

        return await run_in_shared_worker_loop(job)

    # ----- indexing -------------------------------------------------------

    async def initialize(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        self._ensure_available()
        progress_callback = kwargs.get("progress_callback")
        kb_dir = resolve_kb_dir(self.kb_base_dir, kb_name)
        root_dir = resolve_storage_dir_for_rebuild(kb_dir, None)
        self.logger.info(
            "Initializing KB '%s' with %d file(s) using LightRAG", kb_name, len(file_paths)
        )
        try:
            count = await self._run_indexing(
                storage.working_dir(root_dir), file_paths, progress_callback
            )
            if count == 0:
                self.logger.error("LightRAG: no extractable documents for '%s'", kb_name)
                await self._drop_rag(root_dir)
                self._cleanup_failed_version_dir(root_dir)
                return False
            if not storage.has_output(root_dir):
                details = storage.failure_summary(root_dir)
                message = f"LightRAG did not produce a ready index for '{kb_name}'"
                if details:
                    message = f"{message}: {details}"
                self.logger.error(message)
                await self._drop_rag(root_dir)
                self._cleanup_failed_version_dir(root_dir)
                raise RuntimeError(message)
            self._raise_if_corrupt_index(root_dir)
            storage.write_meta(root_dir, storage.read_vector_storage(root_dir))
            self.logger.info("KB '%s' initialized with LightRAG (%d docs)", kb_name, count)
            return True
        except asyncio.CancelledError:
            self._cleanup_failed_version_dir(root_dir)
            raise
        except Exception as exc:
            self.logger.error("Failed to initialize LightRAG KB: %s", exc)
            self.logger.error(traceback.format_exc())
            await self._drop_rag(root_dir)
            self._cleanup_failed_version_dir(root_dir)
            raise

    async def add_documents(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        self._ensure_available()
        progress_callback = kwargs.get("progress_callback")
        kb_dir = resolve_kb_dir(self.kb_base_dir, kb_name)
        existing = resolve_storage_dir_for_read(kb_dir, None)
        if existing is not None and storage.has_output(existing):
            is_update = True
            root_dir = existing
        else:
            is_update = False
            root_dir = resolve_storage_dir_for_rebuild(kb_dir, None)

        self.logger.info(
            "Adding %d document(s) to LightRAG KB '%s' (update=%s)",
            len(file_paths),
            kb_name,
            is_update,
        )
        try:
            count = await self._run_indexing(
                storage.working_dir(root_dir), file_paths, progress_callback
            )
            if count == 0:
                self.logger.warning("LightRAG: no extractable documents to add for '%s'", kb_name)
                return False
            if not storage.has_output(root_dir):
                details = storage.failure_summary(root_dir)
                message = f"LightRAG did not produce a ready index for '{kb_name}'"
                if details:
                    message = f"{message}: {details}"
                self.logger.error(message)
                if not is_update:
                    await self._drop_rag(root_dir)
                    self._cleanup_failed_version_dir(root_dir)
                raise RuntimeError(message)
            self._raise_if_corrupt_index(root_dir)
            storage.write_meta(root_dir, storage.read_vector_storage(root_dir))
            self.logger.info("Added %d doc(s) to LightRAG KB '%s'", count, kb_name)
            return True
        except asyncio.CancelledError:
            if not is_update:
                self._cleanup_failed_version_dir(root_dir)
            raise
        except Exception as exc:
            self.logger.error("Failed to add documents to LightRAG KB: %s", exc)
            self.logger.error(traceback.format_exc())
            if not is_update:
                await self._drop_rag(root_dir)
                self._cleanup_failed_version_dir(root_dir)
            raise

    # ----- retrieval ------------------------------------------------------

    async def search(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
        kb_dir = resolve_kb_dir(self.kb_base_dir, kb_name)
        root_dir = resolve_storage_dir_for_read(kb_dir, None)

        if root_dir is None or not storage.has_output(root_dir):
            return {
                "query": query,
                "answer": (
                    "This LightRAG knowledge base has no index yet. Add documents before querying."
                ),
                "content": "",
                "sources": [],
                "provider": storage.PROVIDER,
                "needs_reindex": True,
            }

        graph_problem = storage.graph_integrity_error(root_dir)
        if graph_problem:
            return {
                "query": query,
                "answer": (
                    "This LightRAG index is corrupted and cannot be queried safely. "
                    f"{graph_problem}. Rebuild the knowledge base before querying."
                ),
                "content": "",
                "sources": [],
                "provider": storage.PROVIDER,
                "needs_reindex": True,
                "error_type": "corrupt_index",
            }

        mode = self._resolve_mode(kb_name, kwargs)
        owner_loop = asyncio.get_running_loop()
        try:
            self._ensure_available()

            async def job(
                io_bridge: OwnerLoopBridge,
            ) -> tuple[str, list[dict[str, Any]]]:
                rag = await self._get_rag(root_dir, io_bridge=io_bridge, owner_loop=owner_loop)
                return await engine.query_with_sources(rag, query, mode)

            answer, sources = await run_in_shared_worker_loop(job)
        except lr_config.LightRagNotAvailableError as exc:
            return self._error_result(query, exc, error_type="not_configured")
        except Exception as exc:
            self.logger.error("LightRAG search failed: %s", exc)
            self.logger.error(traceback.format_exc())
            return self._error_result(query, exc, error_type="retrieval_error")

        return {
            "query": query,
            "answer": answer,
            "content": answer,
            "sources": sources,
            "provider": storage.PROVIDER,
            "mode": mode,
        }

    def _error_result(self, query: str, exc: Exception, *, error_type: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": str(exc),
            "content": "",
            "sources": [],
            "provider": storage.PROVIDER,
            "error_type": error_type,
        }

    # ----- lifecycle ------------------------------------------------------

    async def delete(self, kb_name: str, **kwargs) -> bool:
        kb_dir = resolve_kb_dir(self.kb_base_dir, kb_name)
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
            self.logger.info("Deleted LightRAG KB '%s'", kb_name)
            return True
        return False


__all__ = ["LightRagPipeline"]
