"""Small, bounded query terms for IMA's local retrieval fallbacks.

IMA normally performs the search itself. These terms are only used when the
remote search returns no matches and DeepTutor must rank inventory titles, or
when selecting relevant pages from a downloaded PDF. Chinese text has no
spaces, so treating an entire phrase as one token misses titles/content that
contain punctuation or particles. Bigrams provide a tolerant fallback without
requiring a language-specific segmenter.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}|[\u3400-\u9fff]{2,}")
_MAX_TERMS = 64


def query_terms(query: str) -> list[str]:
    """Return stable, deduplicated Latin tokens and Chinese phrase/bigrams."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        normalized = term.lower()
        if normalized and normalized not in seen and len(terms) < _MAX_TERMS:
            seen.add(normalized)
            terms.append(normalized)

    for token in _TOKEN_RE.findall(query):
        add(token)
        if any("\u3400" <= char <= "\u9fff" for char in token) and len(token) > 2:
            for index in range(len(token) - 1):
                add(token[index : index + 2])
    return terms


__all__ = ["query_terms"]
