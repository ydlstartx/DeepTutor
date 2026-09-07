"""Legacy query compatibility against real native SDK storage and retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

EmbeddingFunc = pytest.importorskip("lightrag.utils").EmbeddingFunc

from deeptutor.services.rag.pipelines.lightrag import engine, legacy_query, pipeline, storage


def _fixture(root: Path, label: str) -> None:
    faiss = pytest.importorskip("faiss")
    root.mkdir(parents=True)
    metadata = {
        "provider": "lightrag",
        "signature": "lightrag",
        "vector_storage": "faiss",
        "version": root.name,
    }
    (root / "meta.json").write_text(json.dumps(metadata))
    chunk = {
        "content": label,
        "tokens": 4,
        "full_doc_id": "doc",
        "file_path": f"{label}.md",
        "chunk_order_index": 0,
    }
    (root / "kv_store_text_chunks.json").write_text(json.dumps({"chunk": chunk}))
    (root / "kv_store_doc_status.json").write_text(
        json.dumps({"doc": {"status": "processed", "chunks_list": ["chunk"]}})
    )
    vector = np.ones((1, 32), dtype=np.float32)
    faiss.normalize_L2(vector)
    for namespace in ("chunks", "entities", "relationships"):
        index = faiss.IndexFlatIP(32)
        index.add(vector if namespace == "chunks" else np.zeros((0, 32), dtype=np.float32))
        file = root / f"faiss_index_{namespace}.index"
        faiss.write_index(index, str(file))
        file.with_name(file.name + ".meta.json").write_text(
            json.dumps({"0": {"__id__": "chunk", **chunk}} if namespace == "chunks" else {})
        )
    nx.write_graphml(nx.Graph(), root / "graph_chunk_entity_relation.graphml")


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def _configure(monkeypatch):
    async def embed(texts, **_kwargs):
        return np.ones((len(texts), 32), dtype=np.float32)

    async def answer(prompt, **_kwargs):
        return "offline answer"

    monkeypatch.setattr(engine, "build_llm_model_func", lambda **_kwargs: answer)
    monkeypatch.setattr(
        engine,
        "build_embedding_func",
        lambda **_kwargs: EmbeddingFunc(embedding_dim=32, max_token_size=8192, func=embed),
    )
    monkeypatch.setattr(engine, "query_kwargs_from_settings", lambda: {})


def test_legacy_faiss_query_isolated_and_never_rewrites_source(monkeypatch, tmp_path):
    _configure(monkeypatch)
    first = tmp_path / "alpha" / "version-1"
    second = tmp_path / "beta" / "version-1"
    _fixture(first, "alpha")
    _fixture(second, "beta")
    before = _snapshot(tmp_path)

    async def query(root):
        with legacy_query.query_view(root) as view:
            rag = engine.build_rag(view, legacy_source=root)
            await engine.initialize(rag)
            try:
                assert all(
                    "__vector__" not in record for record in rag.chunks_vdb._id_to_meta.values()
                )
                assert "chunk" in await rag.chunks_vdb.get_vectors_by_ids(["chunk"])
                result = await rag.chunks_vdb.query("question", top_k=1)
                stored = await rag.text_chunks.get_by_id("chunk")
                answer, sources = await engine.query_with_sources(rag, "question", "naive")
                assert answer == "offline answer"
                assert any(item.get("content") == stored["content"] for item in sources)
                with pytest.raises(RuntimeError, match="query-only"):
                    await rag.text_chunks.upsert({"new": {"content": "must not persist"}})
                assert not rag.enable_llm_cache
                return result[0]["content"], stored["content"], rag.workspace
            finally:
                await engine.finalize(rag, cancel_pending=False)

    async def run():
        return await asyncio.gather(query(first), query(second))

    a, b = asyncio.run(run())
    assert a[:2] == ("alpha", "alpha")
    assert b[:2] == ("beta", "beta")
    assert a[2] != b[2]
    assert _snapshot(tmp_path) == before


def test_legacy_detection_does_not_relabel_failed_native_versions(tmp_path):
    root = tmp_path / "kb" / "version-1"
    _fixture(root, "legacy")
    assert legacy_query.latest_legacy_root(root.parent) == root
    meta = storage._read_meta(root)
    meta["lightrag_adapter_schema"] = storage.ADAPTER_SCHEMA
    (root / "meta.json").write_text(json.dumps(meta))
    assert legacy_query.latest_legacy_root(root.parent) is None


@pytest.mark.asyncio
async def test_legacy_append_still_requires_rebuild(monkeypatch, tmp_path):
    root = tmp_path / "kb" / "version-1"
    _fixture(root, "legacy")
    p = pipeline.LightRagPipeline(kb_base_dir=str(tmp_path))
    with pytest.raises(pipeline.LightRagNeedsReindexError):
        await p.add_documents("kb", ["new.md"])
