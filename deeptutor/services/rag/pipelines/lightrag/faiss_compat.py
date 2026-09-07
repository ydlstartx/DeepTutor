"""Keep the pinned SDK's Faiss loader from duplicating vectors as Python lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def install_lean_loader() -> None:
    """Retain native deferred writes; reconstruct loaded vectors only on demand."""
    import faiss
    from lightrag.kg.faiss_impl import FaissVectorDBStorage

    cls = FaissVectorDBStorage
    if getattr(cls, "_deeptutor_lean", False):
        return
    original_get_vectors = cls.get_vectors_by_ids

    def load(self: Any) -> None:
        index_file = Path(self._faiss_index_file)
        meta_file = Path(self._meta_file)
        if not index_file.exists():
            if meta_file.exists():
                raise ValueError("Faiss metadata exists without its vector index")
            return
        index = faiss.read_index(str(index_file))
        if index.d != self._dim:
            raise ValueError(
                f"Faiss dimension mismatch: stored {index.d}, configured {self._dim}. "
                "Select the original embedding model or rebuild the index."
            )
        records = json.loads(meta_file.read_text(encoding="utf-8"))
        if not isinstance(records, dict):
            raise ValueError("Faiss metadata must be an object")
        metadata = {}
        for key, record in records.items():
            fid = int(key)
            if fid < 0 or fid >= index.ntotal or not isinstance(record, dict):
                raise ValueError("Faiss metadata does not match its vector index")
            metadata[fid] = {k: v for k, v in record.items() if k != "__vector__"}
        if len(metadata) != index.ntotal:
            raise ValueError("Faiss index and metadata have different record counts")
        self._index = index
        self._id_to_meta = metadata

    async def get_vectors(self: Any, ids: list[str]) -> dict[str, list[float]]:
        # The upstream method retains read-your-writes for its pending buffer.
        vectors = await original_get_vectors(self, ids)
        async with self._storage_lock:
            for custom_id in ids:
                if custom_id in vectors:
                    continue
                fid = self._find_faiss_id_by_custom_id(custom_id)
                if fid is not None and fid in self._id_to_meta:
                    vectors[custom_id] = self._index.reconstruct(fid).tolist()
        return vectors

    cls._load_faiss_index = load
    cls.get_vectors_by_ids = get_vectors
    cls._deeptutor_lean = True
