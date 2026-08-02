"""Storage operations for the LlamaIndex RAG pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from deeptutor.services.embedding.validation import validate_embedding_batch
from deeptutor.services.file_io import atomic_write_json, atomic_write_text
from deeptutor.services.rag.index_versioning import (
    EmbeddingSignature,
    find_matching_version,
    resolve_storage_dir_for_read,
    resolve_storage_dir_for_write,
)

from . import ingestion, retrievers, vector_store


@dataclass(frozen=True)
class AddStoragePlan:
    existing_storage: Path | None
    storage_dir: Path


def _storage_path_from_version_entry(entry: dict[str, Any]) -> Path | None:
    storage_path = entry.get("storage_path")
    if storage_path:
        return Path(str(storage_path))

    version_path = entry.get("version_path")
    if not version_path:
        return None

    path = Path(str(version_path))
    layout = str(entry.get("layout") or "")
    if layout == "nested_legacy":
        return path / "llamaindex_storage"
    return path


def cleanup_failed_version_dir(storage_dir: Path) -> bool:
    """Remove an empty flat version dir created by a failed indexing attempt."""
    if not storage_dir.is_dir() or not storage_dir.name.startswith("version-"):
        return False
    storage_empty = not any(child for child in storage_dir.iterdir() if child.name != "meta.json")
    meta_path = storage_dir / "meta.json"
    if storage_empty and not meta_path.exists():
        shutil.rmtree(storage_dir, ignore_errors=True)
        return True
    return False


def resolve_add_storage_plan(kb_dir: Path, signature: EmbeddingSignature | None) -> AddStoragePlan:
    """Choose existing/new storage dirs for incremental adds."""
    matching_version = find_matching_version(kb_dir, signature) if signature is not None else None
    existing_storage = (
        _storage_path_from_version_entry(matching_version) if matching_version else None
    )

    if matching_version and existing_storage and matching_version.get("layout") == "flat":
        return AddStoragePlan(existing_storage=existing_storage, storage_dir=existing_storage)

    if matching_version and existing_storage:
        return AddStoragePlan(
            existing_storage=existing_storage,
            storage_dir=resolve_storage_dir_for_write(kb_dir, signature),
        )

    fallback_storage = resolve_storage_dir_for_read(kb_dir, signature)
    existing_storage = fallback_storage
    fallback_is_flat = (
        fallback_storage is not None
        and fallback_storage.parent == kb_dir
        and fallback_storage.name.startswith("version-")
    )
    storage_dir = (
        fallback_storage if fallback_is_flat else resolve_storage_dir_for_write(kb_dir, signature)
    )
    return AddStoragePlan(existing_storage=existing_storage, storage_dir=storage_dir)


def create_index(documents: list[Any], storage_dir: Path, *, show_progress: bool = True) -> int:
    index, count = ingestion.create_index_from_documents(
        documents, storage_dir, show_progress=show_progress
    )
    retrievers.persist_bm25_retriever(index, storage_dir, top_k=20)
    return count


def insert_documents(existing_storage: Path, storage_dir: Path, documents: list[Any]) -> int:
    index = vector_store.load_index(existing_storage)
    _validate_persisted_embeddings(index, existing_storage)
    if hasattr(index, "insert_nodes"):
        count = ingestion.insert_documents_into_index(index, documents, show_progress=True)
    else:
        # Some tests use a tiny fake index that only implements insert().
        for document in documents:
            index.insert(document)
        count = len(documents)
    index.storage_context.persist(persist_dir=str(storage_dir))
    retrievers.persist_bm25_retriever(index, storage_dir, top_k=20)
    return count


def _validate_embedding_dict(embedding_dict: Any, *, label: str) -> None:
    if not isinstance(embedding_dict, dict) or not embedding_dict:
        return

    validate_embedding_batch(
        list(embedding_dict.values()),
        expected_count=len(embedding_dict),
        binding="llamaindex",
        model=f"persisted-index:{label}",
    )


def _iter_index_embedding_dicts(index: Any):
    """Yield embedding dictionaries exposed by loaded LlamaIndex vector stores."""
    seen: set[int] = set()

    def _yield_store(label: str, vector_store: Any):
        if vector_store is None:
            return
        store_id = id(vector_store)
        if store_id in seen:
            return
        seen.add(store_id)
        data = getattr(vector_store, "data", None)
        embedding_dict = getattr(data, "embedding_dict", None)
        if isinstance(embedding_dict, dict):
            yield label, embedding_dict

    yield from _yield_store("default", getattr(index, "vector_store", None))

    storage_context = getattr(index, "storage_context", None)
    vector_stores = getattr(storage_context, "vector_stores", None)
    if isinstance(vector_stores, dict):
        for namespace, vector_store in vector_stores.items():
            yield from _yield_store(str(namespace), vector_store)


def _embedding_dict_from_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("embedding_dict"), dict):
        return payload["embedding_dict"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("embedding_dict"), dict):
        return data["embedding_dict"]
    return None


def _iter_file_embedding_dicts(storage_dir: Path):
    """Yield embedding dictionaries from persisted SimpleVectorStore JSON files.

    Binary FAISS indexes share the ``*vector_store.json`` filename but are not
    JSON, so they are skipped (their vectors are validated at index build time).
    """
    for path in sorted(storage_dir.glob("*vector_store.json")):
        try:
            with open(path, "rb") as probe:
                if probe.read(1)[:1] != b"{":
                    continue  # binary FAISS index, not a JSON vector store
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        embedding_dict = _embedding_dict_from_payload(payload)
        if isinstance(embedding_dict, dict):
            yield path.name, embedding_dict


def _validate_persisted_embeddings(index: Any, storage_dir: Path | None = None) -> None:
    """Fail early when a persisted vector store contains unusable vectors."""
    try:
        for label, embedding_dict in _iter_index_embedding_dicts(index):
            _validate_embedding_dict(embedding_dict, label=label)
        if storage_dir is not None:
            for label, embedding_dict in _iter_file_embedding_dicts(storage_dir):
                _validate_embedding_dict(embedding_dict, label=label)
    except ValueError as exc:
        raise ValueError(
            "RAG index contains invalid embedding vectors. Re-index the "
            "knowledge base with the current embedding provider/model before "
            f"querying it again. Details: {exc}"
        ) from exc


def validate_storage_embeddings(storage_dir: Path) -> None:
    """Validate persisted vector-store files without running a retrieval."""
    _validate_persisted_embeddings(None, storage_dir)


# Loaded indexes are cached per storage dir so repeated queries never re-read or
# re-validate the (potentially large) persisted store. Entries are keyed by a
# freshness token derived from the store files' mtimes, so a re-index or
# incremental insert naturally invalidates the stale entry.
@dataclass
class _CachedIndex:
    index: Any
    last_used: float


_INDEX_CACHE: "OrderedDict[tuple[str, tuple[int, ...]], _CachedIndex]" = OrderedDict()
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE_MAXSIZE = 2
_INDEX_CACHE_IDLE_SECONDS = 10 * 60


def _prune_index_cache_locked(now: float) -> None:
    stale = [
        key
        for key, entry in _INDEX_CACHE.items()
        if now - entry.last_used >= _INDEX_CACHE_IDLE_SECONDS
    ]
    for key in stale:
        _INDEX_CACHE.pop(key, None)


def _freshness_token(storage_dir: Path) -> tuple[int, ...]:
    token: list[int] = []
    for name in ("docstore.json", vector_store.DEFAULT_VECTOR_STORE_FILENAME):
        try:
            token.append((storage_dir / name).stat().st_mtime_ns)
        except OSError:
            token.append(0)
    return tuple(token)


def _load_validated_index(storage_dir: Path) -> Any:
    """Load an index once and validate its embeddings (cache-miss path)."""
    index = vector_store.load_index(storage_dir)
    _validate_persisted_embeddings(index, storage_dir)
    return index


def _cached_index(storage_dir: Path) -> Any:
    key = (str(storage_dir.resolve()), _freshness_token(storage_dir))
    now = time.monotonic()
    with _INDEX_CACHE_LOCK:
        _prune_index_cache_locked(now)
        entry = _INDEX_CACHE.get(key)
        if entry is not None:
            entry.last_used = now
            _INDEX_CACHE.move_to_end(key)
            return entry.index

    # Load outside the lock so a slow first load of one KB does not block other
    # KBs' queries. A concurrent duplicate load is harmless (idempotent).
    index = _load_validated_index(storage_dir)
    with _INDEX_CACHE_LOCK:
        # Drop any superseded entry for the same storage dir (older token).
        for stale in [existing for existing in _INDEX_CACHE if existing[0] == key[0]]:
            _INDEX_CACHE.pop(stale, None)
        _INDEX_CACHE[key] = _CachedIndex(index=index, last_used=time.monotonic())
        _INDEX_CACHE.move_to_end(key)
        while len(_INDEX_CACHE) > _INDEX_CACHE_MAXSIZE:
            _INDEX_CACHE.popitem(last=False)
    return index


def clear_index_cache() -> None:
    """Drop all cached indexes (used by tests and after destructive edits)."""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


def prune_index_cache() -> int:
    """Drop indexes idle past the single-user warm window."""
    with _INDEX_CACHE_LOCK:
        before = len(_INDEX_CACHE)
        _prune_index_cache_locked(time.monotonic())
        return before - len(_INDEX_CACHE)


def retrieve_nodes(storage_dir: Path, query: str, *, top_k: int = 5) -> list[Any]:
    index = _cached_index(Path(storage_dir))
    retriever = retrievers.build_retriever(index, Path(storage_dir), top_k=top_k)
    return retriever.retrieve(query)


def delete_kb_dir(kb_dir: Path) -> bool:
    if kb_dir.exists():
        shutil.rmtree(kb_dir)
        return True
    return False


def rename_source_references(storage_dir: Path, *, old_path: Path, new_path: Path) -> int:
    """Rewrite source-file metadata after a raw document is renamed.

    Both ``docstore.json`` and the BM25 corpus embed ``file_name`` /
    ``file_path`` in node metadata; leaving them stale breaks citation labels
    and the source-file preview link. ``file_name`` is only retitled for nodes
    whose ``file_path`` matches exactly — same-named files in other folders
    share the basename and must not be touched. Returns the number of records
    patched (0 when the file was never indexed).
    """
    old_file_path = str(old_path)
    new_file_path = str(new_path)
    old_file_name = old_path.name
    new_file_name = new_path.name

    def _patch_metadata(metadata: Any) -> bool:
        if not isinstance(metadata, dict) or metadata.get("file_path") != old_file_path:
            return False
        metadata["file_path"] = new_file_path
        if metadata.get("file_name") == old_file_name:
            metadata["file_name"] = new_file_name
        return True

    patched = 0

    docstore_path = storage_dir / "docstore.json"
    if docstore_path.is_file():
        try:
            payload = json.loads(docstore_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            touched = 0
            data = payload.get("docstore/data")
            if isinstance(data, dict):
                for record in data.values():
                    if not isinstance(record, dict):
                        continue
                    node = record.get("__data__", record)
                    if isinstance(node, dict) and _patch_metadata(node.get("metadata")):
                        touched += 1
            ref_info = payload.get("docstore/ref_doc_info")
            if isinstance(ref_info, dict):
                for entry in ref_info.values():
                    if isinstance(entry, dict) and _patch_metadata(entry.get("metadata")):
                        touched += 1
            if touched:
                atomic_write_json(docstore_path, payload)
                patched += touched

    corpus_path = storage_dir / "bm25_retriever" / "corpus.jsonl"
    if corpus_path.is_file():
        touched = 0
        lines: list[str] = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            row: dict[str, Any] | None = None
            if line.strip():
                try:
                    parsed = json.loads(line)
                    row = parsed if isinstance(parsed, dict) else None
                except Exception:
                    row = None
            if row is None:
                lines.append(line)
                continue
            line_touched = False
            if row.get("file_path") == old_file_path:
                row["file_path"] = new_file_path
                if row.get("file_name") == old_file_name:
                    row["file_name"] = new_file_name
                line_touched = True
            node_content = row.get("_node_content")
            if isinstance(node_content, str):
                try:
                    node = json.loads(node_content)
                except Exception:
                    node = None
                if isinstance(node, dict) and _patch_metadata(node.get("metadata")):
                    row["_node_content"] = json.dumps(node, ensure_ascii=False)
                    line_touched = True
            if line_touched:
                touched += 1
                lines.append(json.dumps(row, ensure_ascii=False))
            else:
                lines.append(line)
        if touched:
            atomic_write_text(corpus_path, "\n".join(lines) + "\n")
            patched += touched

    return patched
