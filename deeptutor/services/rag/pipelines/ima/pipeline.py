"""Retrieval-only pipeline backed by a Tencent IMA knowledge base.

Implements the same contract as the other pipelines (see ``..base.RAGPipeline``)
but owns no index: an ``ima`` KB is a connection pointer (``type: ima`` in
``kb_config.json``) to a library the user keeps in IMA and curates there. Only
:meth:`search` does real work — it resolves the KB's credentials, asks IMA for
matching passages, tops the thin ones up with real source text, and shapes them
for the ``rag`` tool. Documents are added in IMA (or through the IMA capability's
own tools), so :meth:`initialize` / :meth:`add_documents` are not part of this
engine's job and fail with a clear message; :meth:`delete` is a no-op because
deleting the KB only drops DeepTutor's pointer (handled by the manager) and must
never touch the user's IMA library.

The retrieval *policy* — which matches deserve a full-text fetch — lives in
:mod:`.sources`; this module only orchestrates.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import re
from typing import Any, Dict, List, Optional

from deeptutor.runtime.home import get_runtime_data_root
from deeptutor.services.rag.provider_binding import load_kb_config_entry

from . import media as media_ops
from . import sources as source_policy
from .config import ImaNotConfiguredError, resolve_kb_config

logger = logging.getLogger(__name__)

PROVIDER = "ima"
DEFAULT_KB_BASE_DIR = str(get_runtime_data_root() / "knowledge_bases")

# How many matched passages one retrieval feeds into the prompt. IMA returns a
# highlight snippet per item rather than whole documents, so this is a passage
# budget, not a document budget.
_DEFAULT_TOP_K = 10
_MAX_TOP_K = 50
_ZERO_HIT_INVENTORY_LIMIT = 200
_ZERO_HIT_MAX_REQUESTS = 8
_ZERO_HIT_MAX_DEPTH = 3


class ImaPipeline:
    """Query a Tencent IMA knowledge base on behalf of a connected KB."""

    def __init__(self, kb_base_dir: Optional[str] = None, *, client_factory=None, **_: Any) -> None:
        self.logger = logging.getLogger(__name__)
        self.kb_base_dir = kb_base_dir or DEFAULT_KB_BASE_DIR
        # Injection seam for tests: (config) -> client. None uses the real client.
        self._client_factory = client_factory

    # ----- helpers --------------------------------------------------------

    def _client(self, config):
        if self._client_factory is not None:
            return self._client_factory(config)
        from .client import ImaClient

        return ImaClient(config)

    @staticmethod
    def _top_k(kwargs: dict[str, Any]) -> int:
        try:
            requested = int(kwargs.get("top_k") or _DEFAULT_TOP_K)
        except (TypeError, ValueError):
            return _DEFAULT_TOP_K
        return max(1, min(requested, _MAX_TOP_K))

    # ----- retrieval ------------------------------------------------------

    async def search(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
        try:
            config = resolve_kb_config(load_kb_config_entry(self.kb_base_dir, kb_name))
        except ImaNotConfiguredError as exc:
            return self._error_result(query, exc, error_type="not_configured")

        try:
            client = self._client(config)
            page = await client.search_knowledge(query, limit=self._top_k(kwargs))
        except Exception as exc:
            self.logger.error("IMA search failed for '%s': %s", kb_name, exc)
            return self._error_result(query, exc, error_type="retrieval_error")

        sources = source_policy.documents_to_sources(page.documents)
        diagnostic = ""
        if not sources:
            sources, diagnostic = await self._zero_hit_fallback(client, query)
        await self._hydrate(client, sources, query=query)
        content = source_policy.render_context(sources)
        if diagnostic and not any(source.get("content") for source in sources):
            content = f"{diagnostic}\n\n{content}".strip()
        elif not content and diagnostic:
            content = diagnostic
        return {
            "query": query,
            "answer": content,
            "content": content,
            "sources": sources,
            "provider": PROVIDER,
            **({"retrieval_diagnostic": diagnostic} if diagnostic else {}),
        }

    async def _zero_hit_fallback(self, client, query: str) -> tuple[list[dict[str, Any]], str]:
        """Use a bounded inventory walk when IMA search returns no matches."""
        documents = []
        queue: deque[tuple[str, int]] = deque([("", 0)])
        seen_folders: set[str] = set()
        requests = 0
        truncated = False
        try:
            while queue and len(documents) < _ZERO_HIT_INVENTORY_LIMIT:
                folder_id, depth = queue.popleft()
                cursor = ""
                while True:
                    if requests >= _ZERO_HIT_MAX_REQUESTS:
                        truncated = True
                        break
                    page = await client.get_knowledge_list(
                        folder_id=folder_id,
                        cursor=cursor,
                        limit=50,
                    )
                    requests += 1
                    remaining = _ZERO_HIT_INVENTORY_LIMIT - len(documents)
                    documents.extend(page.documents[:remaining])
                    if len(documents) >= _ZERO_HIT_INVENTORY_LIMIT:
                        truncated = True
                        break
                    for folder in page.folders:
                        if folder.folder_id in seen_folders or folder.folder_id == folder_id:
                            continue
                        seen_folders.add(folder.folder_id)
                        if depth >= _ZERO_HIT_MAX_DEPTH:
                            truncated = True
                        else:
                            queue.append((folder.folder_id, depth + 1))
                    cursor = page.next_cursor
                    if page.is_end or not cursor:
                        break
                if truncated and requests >= _ZERO_HIT_MAX_REQUESTS:
                    break
        except Exception as exc:
            self.logger.warning(
                "Could not list IMA contents after a zero-hit search (%s)",
                type(exc).__name__,
            )
            return [], (
                "IMA returned no content matches, and DeepTutor could not read the remote "
                "file inventory to diagnose the empty result."
            )

        if not documents:
            return [], "IMA returned no content matches and the remote library exposes no files."

        ranked = sorted(
            ((_title_match_score(query, document.title), document) for document in documents),
            key=lambda pair: pair[0],
            reverse=True,
        )
        candidates = [document for score, document in ranked if score > 0][
            : source_policy.DEFAULT_HYDRATION_BUDGET
        ]
        if not candidates and len(documents) == 1:
            candidates = documents

        total_label = f"at least {len(documents)}" if truncated or queue else str(len(documents))
        if not candidates:
            return [], (
                "IMA returned no content matches even though the remote library exposes "
                f"{total_label} files. No file name matched the query, so DeepTutor did not "
                "download unrelated documents."
            )

        return source_policy.documents_to_sources(candidates), (
            "IMA returned no content matches. DeepTutor fell back to reading "
            f"{len(candidates)} remote file(s) from a library exposing {total_label} file(s)."
        )

    async def _hydrate(
        self,
        client,
        sources: list[dict[str, Any]],
        *,
        query: str = "",
    ) -> None:
        """Replace thin or missing snippets with real source text, concurrently.

        Each fetch is independent, so they run together — a search that needs
        four documents costs one round-trip's latency, not four. A document that
        cannot be loaded keeps its snippet (or stays a title-only reference):
        one unavailable file must never discard the other matches.
        """
        targets = source_policy.hydration_targets(sources)
        if not targets:
            return
        results = await asyncio.gather(
            *(self._fetch_text(client, sources[index], query=query) for index in targets),
            return_exceptions=True,
        )
        for index, result in zip(targets, results):
            if isinstance(result, BaseException):
                # HTTP errors may embed a signed COS URL; log only the error
                # class so short-lived download credentials never reach logs.
                self.logger.warning(
                    "Could not load IMA media '%s' (%s)",
                    sources[index]["chunk_id"],
                    type(result).__name__,
                )
            elif result:
                sources[index]["content"] = result

    @staticmethod
    async def _fetch_text(client, source: dict[str, Any], *, query: str = "") -> str:
        media = await client.get_media_content(source["chunk_id"])
        return await media_ops.extract_text(
            media,
            source["title"],
            max_chars=source_policy.MAX_FULLTEXT_CHARS,
            query=query,
        )

    def _error_result(self, query: str, exc: Exception, *, error_type: str) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": str(exc),
            "content": "",
            "sources": [],
            "provider": PROVIDER,
            "error_type": error_type,
        }

    # ----- indexing (not applicable — owned by IMA) ------------------------

    async def initialize(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        raise RuntimeError(
            "Tencent IMA knowledge bases are indexed by IMA; DeepTutor does not "
            "build or store their index. Add documents in IMA directly."
        )

    async def add_documents(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
        return await self.initialize(kb_name, file_paths, **kwargs)

    # ----- lifecycle ------------------------------------------------------

    async def delete(self, kb_name: str, **kwargs) -> bool:
        # The KB is only a pointer; the manager removes its config entry. Never
        # touch the user's IMA library. Nothing local to clean up here.
        return True


def _query_terms(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", query)]


def _title_match_score(query: str, title: str) -> int:
    lowered = title.lower()
    return sum(term in lowered for term in _query_terms(query))


__all__ = ["ImaPipeline", "PROVIDER"]
