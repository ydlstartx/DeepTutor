"""Read-only access to published pre-native LightRAG indexes.

Only queries enter this adapter. Native ingestion still requires a native
published version, so a legacy index can never receive new-format writes.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Any, Iterator

from . import storage

_STORE_GLOBS = (
    "kv_store_*.json",
    "vdb_*.json",
    "faiss_index_*.index",
    "faiss_index_*.index.meta.json",
    "graph_*.graphml",
)
_STORAGE_ATTRIBUTES = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)


def latest_legacy_root(kb_dir: Path) -> Path | None:
    from deeptutor.services.rag.index_versioning import list_kb_versions

    for item in list_kb_versions(kb_dir):
        root = Path(str(item.get("storage_path") or ""))
        meta = storage._read_meta(root)
        if (
            meta
            and meta.get("provider") == "lightrag"
            and meta.get("signature") == "lightrag"
            and meta.get("lightrag_adapter_schema") is None
            and storage.has_output(root)
            and not storage.graph_integrity_error(root)
        ):
            return root
    return None


@contextmanager
def query_view(source: Path) -> Iterator[Path]:
    """Expose flat legacy stores under the SDK's isolated workspace layout.

    Constructors may clean temporary files or create auxiliary paths; those
    operations stay in this temporary view. Store mutation methods are disabled
    before initialization, and caches are disabled by the engine constructor.
    The data is linked rather than copied, so multi-GB Faiss indexes need no
    second disk copy. No metadata or raw documents are exposed in this view.
    """
    from .engine import workspace_for

    source = source.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="deeptutor-lightrag-query-") as directory:
        view = Path(directory)
        workspace = view / workspace_for(source)
        workspace.mkdir()
        for pattern in _STORE_GLOBS:
            for file in source.glob(pattern):
                if file.is_symlink() or not file.is_file():
                    raise ValueError(f"Legacy LightRAG store must be a regular file: {file.name}")
                (workspace / file.name).symlink_to(file)
        yield view


async def _reject_write(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Legacy LightRAG indexes are query-only; rebuild before modifying them.")


async def _no_flush(*_args: Any, **_kwargs: Any) -> bool:
    return True


def make_read_only(rag: Any) -> None:
    """Prevent SDK cache, migration, ingestion, and finalizer writes."""
    for name in _STORAGE_ATTRIBUTES:
        store = getattr(rag, name, None)
        if store is None:
            continue
        for method in (
            "upsert",
            "delete",
            "drop",
            "upsert_node",
            "upsert_edge",
            "remove_nodes",
            "remove_edges",
            "delete_entity",
            "delete_entity_relation",
            "delete_relation",
        ):
            if hasattr(store, method):
                setattr(store, method, _reject_write)
        store.index_done_callback = _no_flush
        store.finalize = _no_flush
