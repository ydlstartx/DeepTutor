"""Durable, content-addressed cache for image descriptions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from deeptutor.services.file_io import atomic_write_json, exclusive_write_lock

_CACHE_VERSION = 1


def description_signature(*, binding: str, model: str, system_prompt: str, prompt: str) -> str:
    """Fingerprint every input that can materially change a description."""
    payload = {
        "version": _CACHE_VERSION,
        "binding": binding.strip().lower(),
        "model": model.strip(),
        "system_prompt": system_prompt,
        "prompt": prompt,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ImageDescriptionCache:
    """Store one small JSON record per image/model/prompt combination."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, image_hash: str, signature: str) -> Path:
        return self.root / image_hash[:2] / image_hash / f"{signature}.json"

    def get(self, *, image_hash: str, signature: str) -> str | None:
        path = self._path(image_hash, signature)
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if (
            payload.get("version") != _CACHE_VERSION
            or payload.get("image_hash") != image_hash
            or payload.get("signature") != signature
        ):
            return None
        description = payload.get("description")
        if not isinstance(description, str) or not description.strip():
            return None
        return description.strip()

    def put(
        self,
        *,
        image_hash: str,
        signature: str,
        binding: str,
        model: str,
        description: str,
    ) -> None:
        normalized = description.strip()
        if not normalized:
            return
        path = self._path(image_hash, signature)
        with exclusive_write_lock(path):
            # Preserve a valid record another worker may have completed first.
            if self.get(image_hash=image_hash, signature=signature) is not None:
                return
            atomic_write_json(
                path,
                {
                    "version": _CACHE_VERSION,
                    "image_hash": image_hash,
                    "signature": signature,
                    "binding": binding,
                    "model": model,
                    "description": normalized,
                },
            )


__all__ = ["ImageDescriptionCache", "description_signature"]
