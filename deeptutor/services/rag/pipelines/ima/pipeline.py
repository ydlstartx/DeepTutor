"""Retrieval-only pipeline backed by a Tencent IMA knowledge base.

Implements the same contract as the other pipelines (see ``..base.RAGPipeline``)
but owns no index: an ``ima`` KB is a connection pointer (``type: ima`` in
``kb_config.json``) to a library the user keeps in IMA and curates there. Only
:meth:`search` does real work — it resolves the KB's credentials, asks IMA for
matching passages, and shapes them for the ``rag`` tool. Documents are added in
IMA, so :meth:`initialize` / :meth:`add_documents` are not part of this engine's
job and fail with a clear message; :meth:`delete` is a no-op because deleting the
KB only drops DeepTutor's pointer (handled by the manager) and must never touch
the user's IMA library.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import logging
from pathlib import PurePath
import re
from typing import Any, Dict, List, Optional

from deeptutor.runtime.home import get_runtime_data_root
from deeptutor.services.rag.provider_binding import load_kb_config_entry
from deeptutor.utils.document_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text_from_bytes,
    extract_text_from_path,
)

from .client import MAX_MEDIA_BYTES, MAX_PDF_MEDIA_BYTES, ImaMediaContent
from .config import ImaNotConfiguredError, resolve_kb_config

logger = logging.getLogger(__name__)

PROVIDER = "ima"
DEFAULT_KB_BASE_DIR = str(get_runtime_data_root() / "knowledge_bases")

# How many matched passages one retrieval feeds into the prompt. IMA returns a
# highlight snippet per item rather than whole documents, so this is a passage
# budget, not a document budget.
_DEFAULT_TOP_K = 10

# Full-document retrieval is only a fallback for title-only IMA matches. Keep
# its network and prompt footprint predictable even when ``top_k`` is large.
_MAX_FULLTEXT_ITEMS = 3
_MAX_FULLTEXT_CHARS = 12_000
_ZERO_HIT_INVENTORY_LIMIT = 200


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
        return max(1, min(requested, 50))

    # ----- retrieval ------------------------------------------------------

    async def search(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
        try:
            config = resolve_kb_config(load_kb_config_entry(self.kb_base_dir, kb_name))
        except ImaNotConfiguredError as exc:
            return self._error_result(query, exc, error_type="not_configured")

        try:
            client = self._client(config)
            items = await client.search_knowledge(query, limit=self._top_k(kwargs))
        except Exception as exc:
            self.logger.error("IMA search failed for '%s': %s", kb_name, exc)
            return self._error_result(query, exc, error_type="retrieval_error")

        diagnostic = ""
        if not items:
            items, diagnostic = await self._zero_hit_fallback(client, query)

        sources = _sources_from_items(items)
        await self._hydrate_title_only_sources(client, sources, query=query)
        content = _render_context(sources)
        if diagnostic and not any(source["content"] for source in sources):
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

    async def _zero_hit_fallback(self, client, query: str) -> tuple[list[dict], str]:
        """Read a small, unambiguous remote file set when IMA search is empty."""
        try:
            tree = await client.list_knowledge_tree(max_items=_ZERO_HIT_INVENTORY_LIMIT)
        except Exception as exc:
            self.logger.warning(
                "Could not list IMA contents after a zero-hit search (%s)", type(exc).__name__
            )
            return [], (
                "IMA returned no content matches, and DeepTutor could not read the remote "
                "file inventory to diagnose the empty result."
            )

        files = [item for item in tree.get("items", []) if item.get("type") == "file"]
        if not files:
            return [], "IMA returned no content matches and the remote library exposes no files."

        ranked = sorted(
            ((_title_match_score(query, str(item.get("name") or "")), item) for item in files),
            key=lambda pair: pair[0],
            reverse=True,
        )
        candidates = [item for score, item in ranked if score > 0][:_MAX_FULLTEXT_ITEMS]
        if not candidates and len(files) == 1:
            candidates = files

        total = len(files)
        truncated = bool(tree.get("truncated"))
        total_label = f"at least {total}" if truncated else str(total)
        if not candidates:
            return [], (
                f"IMA returned no content matches even though the remote library exposes "
                f"{total_label} files. No file name matched the query, so DeepTutor did not "
                "download unrelated documents."
            )

        fallback_items = [
            {
                "media_id": str(item.get("media_id") or ""),
                "title": str(item.get("path") or item.get("name") or ""),
                "highlight_content": "",
            }
            for item in candidates
        ]
        return fallback_items, (
            f"IMA returned no content matches. DeepTutor fell back to reading "
            f"{len(fallback_items)} remote file(s) from a library exposing {total_label} file(s)."
        )

    async def _hydrate_title_only_sources(
        self,
        client,
        sources: list[dict[str, Any]],
        *,
        query: str = "",
    ) -> None:
        remaining = _MAX_FULLTEXT_ITEMS
        for source in sources:
            if remaining == 0:
                break
            if source["content"] or not source["chunk_id"]:
                continue
            remaining -= 1
            media = None
            try:
                media = await client.get_media_content(source["chunk_id"])
                source["content"] = await _extract_media_text(
                    media,
                    source["title"],
                    query=query,
                )
            except Exception as exc:
                # Search results remain useful as title-only references. One
                # unavailable document must not discard the other matches.
                # HTTP errors may contain a signed COS URL; log only the error
                # class so short-lived download credentials never reach logs.
                self.logger.warning(
                    "Could not load IMA media '%s' (%s)",
                    source["chunk_id"],
                    type(exc).__name__,
                )
            finally:
                if media is not None:
                    media.cleanup()

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


def _sources_from_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map IMA search items into DeepTutor's ``sources`` shape.

    ``highlight_content`` is the matched snippet IMA returns; items whose match
    was on the title alone carry none and are still listed, so the model can see
    the document exists even without a quotable passage.
    """
    sources: list[dict[str, Any]] = []
    for item in items:
        title = str(item.get("title") or "").strip()
        media_id = str(item.get("media_id") or "").strip()
        if not title and not media_id:
            continue
        sources.append(
            {
                "title": title or media_id,
                "content": str(item.get("highlight_content") or "").strip(),
                "source": title or media_id,
                "chunk_id": media_id,
            }
        )
    return sources


async def _extract_media_text(
    media: ImaMediaContent | None,
    title: str,
    *,
    query: str = "",
) -> str:
    if media is None:
        return ""
    if media.text:
        return media.text[:_MAX_FULLTEXT_CHARS].strip()
    if media.local_path:
        filename = _extractable_filename(title, media.filename, media.local_path)
        if not filename:
            return ""
        return await asyncio.to_thread(
            _extract_downloaded_path,
            media.local_path,
            filename,
            query,
        )
    if not media.data:
        return ""

    filename = _extractable_filename(title, media.filename)
    if not filename:
        return ""
    return await asyncio.to_thread(
        extract_text_from_bytes,
        filename,
        media.data,
        max_bytes=MAX_MEDIA_BYTES,
        max_chars=_MAX_FULLTEXT_CHARS,
    )


def _extract_downloaded_path(path: str, filename: str, query: str) -> str:
    if PurePath(filename).suffix.lower() == ".pdf":
        return _extract_relevant_pdf_pages(path, query)
    return extract_text_from_path(
        path,
        max_bytes=MAX_PDF_MEDIA_BYTES,
        max_chars=_MAX_FULLTEXT_CHARS,
    )


def _extract_relevant_pdf_pages(path: str, query: str) -> str:
    try:
        import fitz
    except ImportError:
        return _extract_relevant_pdf_pages_pypdf(path, query)

    with fitz.open(path) as document:
        pages = (
            (page_number, (page.get_text() or "").strip())
            for page_number, page in enumerate(document, start=1)
        )
        return _select_relevant_pages(pages, query)


def _extract_relevant_pdf_pages_pypdf(path: str, query: str) -> str:
    from pypdf import PdfReader

    document = PdfReader(path)
    pages = (
        (page_number, (page.extract_text() or "").strip())
        for page_number, page in enumerate(document.pages, start=1)
    )
    return _select_relevant_pages(pages, query)


def _select_relevant_pages(pages: Iterable[tuple[int, str]], query: str) -> str:
    """Select a bounded set from a streaming ``(page number, text)`` iterator."""

    terms = _query_terms(query)
    leading: list[str] = []
    leading_chars = 0
    # A common query term can occur on every page of a large book. Retain only
    # a small candidate set while scanning so memory stays bounded too.
    max_candidates = 32
    matches: list[tuple[int, int, str]] = []
    for page_number, text in pages:
        if not text:
            continue
        block = f"--- Page {page_number} ---\n{text}"
        if leading_chars < _MAX_FULLTEXT_CHARS:
            retained = block[: _MAX_FULLTEXT_CHARS - leading_chars]
            leading.append(retained)
            leading_chars += len(retained)
        lowered = text.lower()
        score = sum(term in lowered for term in terms)
        if score:
            candidate = (score, page_number, block[:_MAX_FULLTEXT_CHARS])
            if len(matches) < max_candidates:
                matches.append(candidate)
            else:
                weakest = min(
                    range(len(matches)),
                    key=lambda index: (matches[index][0], -matches[index][1]),
                )
                if (score, -page_number) > (
                    matches[weakest][0],
                    -matches[weakest][1],
                ):
                    matches[weakest] = candidate

    if matches:
        selected = sorted(matches, key=lambda item: (-item[0], item[1]))
        return _join_bounded([block for _, _, block in selected])
    return _join_bounded(leading)


def _join_bounded(blocks: list[str]) -> str:
    output: list[str] = []
    size = 0
    for block in blocks:
        separator_size = 2 if output else 0
        remaining = _MAX_FULLTEXT_CHARS - size - separator_size
        if remaining <= 0:
            break
        output.append(block[:remaining])
        size += separator_size + min(len(block), remaining)
    return "\n\n".join(output).strip()


def _query_terms(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", query)]


def _title_match_score(query: str, title: str) -> int:
    lowered = title.lower()
    return sum(term in lowered for term in _query_terms(query))


def _extractable_filename(*candidates: str) -> str:
    for candidate in candidates:
        if PurePath(candidate).suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
            return candidate
    return ""


def _render_context(sources: list[dict[str, Any]]) -> str:
    """Flatten retrieved snippets into the grounded context block."""
    blocks = [
        f"[{index}] {source['title']}\n{source['content']}".rstrip()
        for index, source in enumerate(sources, start=1)
    ]
    return "\n\n".join(blocks)


__all__ = ["ImaPipeline", "PROVIDER"]
