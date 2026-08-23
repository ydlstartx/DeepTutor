"""Fetching and reading one IMA item's actual content.

An IMA knowledge item is either a *note* (whose text the notes API returns
directly) or a *file* (which IMA hands over as a short-lived Tencent COS
download URL). This module owns the file half — the download and the text
extraction — so both retrieval (``pipeline``) and the capability's ``ima_read``
tool read documents through exactly one implementation.

Security boundary
-----------------
The download URL arrives inside an API response, so it is treated as untrusted
input: it must be HTTPS and inside Tencent COS (:data:`_COS_ROOT_DOMAIN`), which
prevents a tampered or buggy response from turning retrieval into an SSRF
primitive. Redirects are not followed, IMA's own credentials are never attached
to this separate client, and hop-by-hop / identity headers are stripped from
whatever header set the response asked us to send. Size is capped
(:data:`MAX_MEDIA_BYTES`) both by the advertised ``content-length`` and while
streaming, so a mis-sized response cannot exhaust memory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path, PurePath, PurePosixPath
import re
import tempfile
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

from deeptutor.utils.document_extractor import (
    SUPPORTED_DOC_EXTENSIONS,
    extract_text_from_bytes,
    extract_text_from_path,
)

from .envelope import ImaAPIError

# Official IMA media links are short-lived Tencent COS URLs; the downloader
# refuses anything outside that boundary.
_COS_ROOT_DOMAIN = "myqcloud.com"
_IMA_MEDIA_HOSTS = frozenset({"res-skb.ima.qq.com"})

# Headers we never forward, whatever the API response asks for: hop-by-hop
# fields and anything that would carry identity to COS.
_FORBIDDEN_MEDIA_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "cookie",
        "host",
        "proxy-authorization",
        "transfer-encoding",
    }
)

MAX_MEDIA_BYTES = 20 * 1024 * 1024
MAX_PDF_MEDIA_BYTES = 200 * 1024 * 1024

logger = logging.getLogger(__name__)

_CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/json": ".json",
    "text/csv": ".csv",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
}


@dataclass(frozen=True)
class ImaMediaContent:
    """One IMA item's content, as either note text or downloaded file bytes."""

    text: str = ""
    data: bytes = b""
    filename: str = ""
    local_path: str = ""

    def cleanup(self) -> None:
        """Remove a streamed temporary download, if this content owns one."""
        if not self.local_path:
            return
        try:
            Path(self.local_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary IMA media file")


async def download_media(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> ImaMediaContent:
    """Download IMA media with a 20 MB memory / 200 MB streamed-PDF cap."""
    validate_media_url(url)
    async with httpx.AsyncClient(
        timeout=timeout,
        transport=transport,
        follow_redirects=False,
    ) as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            filename = media_filename(url, content_type)
            max_bytes = _download_limit(filename, content_type)
            length = response.headers.get("content-length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise ImaAPIError(_limit_message(max_bytes))

            if max_bytes == MAX_PDF_MEDIA_BYTES:
                suffix = PurePosixPath(filename).suffix or ".pdf"
                temp_path = ""
                try:
                    with tempfile.NamedTemporaryFile(
                        prefix="deeptutor-ima-",
                        suffix=suffix,
                        delete=False,
                    ) as handle:
                        temp_path = handle.name
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise ImaAPIError(_limit_message(max_bytes))
                            handle.write(chunk)
                    return ImaMediaContent(filename=filename, local_path=temp_path)
                except Exception:
                    if temp_path:
                        Path(temp_path).unlink(missing_ok=True)
                    raise

            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ImaAPIError(_limit_message(max_bytes))
            return ImaMediaContent(data=bytes(body), filename=filename)


async def extract_text(
    media: ImaMediaContent | None,
    *title_candidates: str,
    max_chars: int,
    query: str = "",
) -> str:
    """Plain text for *media*, truncated to *max_chars* (``""`` when unreadable).

    Note text is already plain. File bytes are decoded by
    :func:`~deeptutor.utils.document_extractor.extract_text_from_bytes`, which
    needs a filename to pick a decoder — the download's own filename or the
    item title, whichever carries a supported extension. Extraction is CPU-bound
    (PDF / Office parsing), so it runs in a worker thread.
    """
    if media is None:
        return ""
    try:
        if media.text:
            return media.text[:max_chars].strip()
        filename = _extractable_filename(media.filename, *title_candidates)
        if not filename:
            return ""
        if media.local_path:
            return await asyncio.to_thread(
                _extract_downloaded_path,
                media.local_path,
                filename,
                query,
                max_chars,
            )
        if not media.data:
            return ""
        return await asyncio.to_thread(
            extract_text_from_bytes,
            filename,
            media.data,
            max_bytes=MAX_MEDIA_BYTES,
            max_chars=max_chars,
        )
    finally:
        media.cleanup()


def validate_media_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if parsed.scheme != "https" or not hostname:
        raise ImaAPIError("IMA media URL must use HTTPS.")
    is_cos = hostname == _COS_ROOT_DOMAIN or hostname.endswith(f".{_COS_ROOT_DOMAIN}")
    if not is_cos and hostname not in _IMA_MEDIA_HOSTS:
        raise ImaAPIError("IMA media URL is outside Tencent COS.")


def media_headers(raw: object) -> dict[str, str]:
    """The subset of a response's suggested download headers we will forward."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).lower() not in _FORBIDDEN_MEDIA_HEADERS and isinstance(value, (str, int, float))
    }


def media_filename(url: str, content_type: str | None) -> str:
    """A filename for the download, inferring an extension when the path lacks one."""
    name = unquote(PurePosixPath(urlparse(url).path).name).strip()
    if "." in name:
        return name
    media_type = str(content_type or "").partition(";")[0].strip().lower()
    return f"{name or 'ima-document'}{_CONTENT_TYPE_EXTENSIONS.get(media_type, '')}"


def _download_limit(filename: str, content_type: str | None) -> int:
    media_type = str(content_type or "").partition(";")[0].strip().lower()
    if PurePosixPath(filename).suffix.lower() == ".pdf" or media_type == "application/pdf":
        return MAX_PDF_MEDIA_BYTES
    return MAX_MEDIA_BYTES


def _limit_message(max_bytes: int) -> str:
    return f"IMA media exceeds the {max_bytes // (1024 * 1024)} MB retrieval limit."


def _extract_downloaded_path(path: str, filename: str, query: str, max_chars: int) -> str:
    if PurePath(filename).suffix.lower() == ".pdf":
        return _extract_relevant_pdf_pages(path, query, max_chars)
    return extract_text_from_path(path, max_bytes=MAX_PDF_MEDIA_BYTES, max_chars=max_chars)


def _extract_relevant_pdf_pages(path: str, query: str, max_chars: int) -> str:
    try:
        import fitz
    except ImportError:
        from pypdf import PdfReader

        document = PdfReader(path)
        pages = (
            (page_number, (page.extract_text() or "").strip())
            for page_number, page in enumerate(document.pages, start=1)
        )
        return _select_relevant_pages(pages, query, max_chars)

    with fitz.open(path) as document:
        pages = (
            (page_number, (page.get_text() or "").strip())
            for page_number, page in enumerate(document, start=1)
        )
        return _select_relevant_pages(pages, query, max_chars)


def _select_relevant_pages(pages: Iterable[tuple[int, str]], query: str, max_chars: int) -> str:
    terms = _query_terms(query)
    leading: list[str] = []
    leading_chars = 0
    max_candidates = 32
    matches: list[tuple[int, int, str]] = []
    for page_number, text in pages:
        if not text:
            continue
        block = f"--- Page {page_number} ---\n{text}"
        if leading_chars < max_chars:
            retained = block[: max_chars - leading_chars]
            leading.append(retained)
            leading_chars += len(retained)
        lowered = text.lower()
        score = sum(term in lowered for term in terms)
        if not score:
            continue
        candidate = (score, page_number, block[:max_chars])
        if len(matches) < max_candidates:
            matches.append(candidate)
            continue
        weakest = min(
            range(len(matches)), key=lambda index: (matches[index][0], -matches[index][1])
        )
        if (score, -page_number) > (matches[weakest][0], -matches[weakest][1]):
            matches[weakest] = candidate

    blocks = (
        [block for _, _, block in sorted(matches, key=lambda item: (-item[0], item[1]))]
        if matches
        else leading
    )
    output: list[str] = []
    size = 0
    for block in blocks:
        separator_size = 2 if output else 0
        remaining = max_chars - size - separator_size
        if remaining <= 0:
            break
        output.append(block[:remaining])
        size += separator_size + min(len(block), remaining)
    return "\n\n".join(output).strip()


def _query_terms(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", query)]


def _extractable_filename(*candidates: str) -> str:
    for candidate in candidates:
        if candidate and PurePath(candidate).suffix.lower() in SUPPORTED_DOC_EXTENSIONS:
            return candidate
    return ""


__all__ = [
    "MAX_MEDIA_BYTES",
    "MAX_PDF_MEDIA_BYTES",
    "ImaMediaContent",
    "download_media",
    "extract_text",
    "media_filename",
    "media_headers",
    "validate_media_url",
]
