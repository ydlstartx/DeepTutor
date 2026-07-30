from __future__ import annotations

import json
from pathlib import Path

from deeptutor.services.rag.index_probe import (
    has_ready_provider_index,
    inspect_kb_versions,
    inspect_provider_index,
    inspect_provider_version,
    provider_failure_summary,
)
from deeptutor.services.rag.pipelines.graphrag import storage as graphrag_storage
from deeptutor.services.rag.pipelines.pageindex import storage as pageindex_storage


def _write_meta(version_dir: Path, *, provider: str, signature: str | None = None) -> None:
    (version_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": version_dir.name,
                "provider": provider,
                "signature": signature or provider,
                "layout": "flat",
            }
        ),
        encoding="utf-8",
    )


def test_llamaindex_requires_real_storage_files(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    (version_dir / "docstore.json").write_text(
        json.dumps({"docstore/data": {"doc-1": {}}}),
        encoding="utf-8",
    )

    probe = inspect_provider_index("llamaindex", version_dir)

    assert probe.ready is False
    assert "index_store.json" in probe.failure_summary
    assert probe.doc_count == 1

    (version_dir / "index_store.json").write_text("{}", encoding="utf-8")
    probe = inspect_provider_index("llamaindex", version_dir)
    assert probe.ready is True
    assert probe.doc_count == 1


def test_llamaindex_doc_count_is_cached_until_docstore_changes(tmp_path: Path) -> None:
    """Repeated probes must not re-parse a large docstore.json; a rewrite
    (new mtime/size) invalidates the cache so counts stay correct."""
    import deeptutor.services.rag.index_probe as probe_module

    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    docstore = version_dir / "docstore.json"
    docstore.write_text(json.dumps({"docstore/data": {"doc-1": {}}}), encoding="utf-8")
    (version_dir / "index_store.json").write_text("{}", encoding="utf-8")

    calls = 0
    real_read_json = probe_module._read_json

    def counting_read_json(path):
        nonlocal calls
        calls += 1
        return real_read_json(path)

    probe_module._JSON_PARSE_CACHE.clear()
    try:
        probe_module._read_json = counting_read_json
        first = inspect_provider_index("llamaindex", version_dir)
        second = inspect_provider_index("llamaindex", version_dir)
        assert first.doc_count == 1 and second.doc_count == 1
        assert calls == 1

        docstore.write_text(
            json.dumps({"docstore/data": {"doc-1": {}, "doc-2": {}}}),
            encoding="utf-8",
        )
        third = inspect_provider_index("llamaindex", version_dir)
        assert third.doc_count == 2
        assert calls == 2
    finally:
        probe_module._read_json = real_read_json
        probe_module._JSON_PARSE_CACHE.clear()


def test_kb_versions_overrule_fake_llamaindex_ready_marker(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    (version_dir / "docstore.json").write_text("{}", encoding="utf-8")
    _write_meta(version_dir, provider="llamaindex", signature="sig")

    versions = inspect_kb_versions(tmp_path, "llamaindex")

    assert versions[0]["ready"] is False
    assert "index_store.json" in versions[0]["failure_summary"]
    assert has_ready_provider_index(tmp_path, "llamaindex") is False
    assert "index_store.json" in provider_failure_summary(tmp_path, "llamaindex")


def test_pageindex_ready_requires_doc_ids(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    _write_meta(version_dir, provider="pageindex")

    probe = inspect_provider_index("pageindex", version_dir)
    assert probe.ready is False

    manifest = pageindex_storage.read_manifest(version_dir)
    pageindex_storage.upsert_doc(manifest, "lesson.pdf", "doc-123")
    pageindex_storage.write_manifest(version_dir, manifest)

    probe = inspect_provider_index("pageindex", version_dir)
    assert probe.ready is True
    assert probe.doc_count == 1


def test_graphrag_ready_requires_core_output_table(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    _write_meta(version_dir, provider="graphrag")

    probe = inspect_provider_index("graphrag", version_dir)
    assert probe.ready is False
    assert "parquet" in probe.failure_summary

    out = graphrag_storage.output_dir(version_dir)
    out.mkdir()
    (out / "entities.parquet").write_bytes(b"placeholder")

    probe = inspect_provider_index("graphrag", version_dir)
    assert probe.ready is True
    assert probe.diagnostics["output_tables"] == ["entities"]


def test_lightrag_uses_doc_status_as_truth(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    _write_meta(version_dir, provider="lightrag")
    (version_dir / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "status": "failed",
                    "file_path": "bad.docx",
                    "error_msg": "parse failed",
                    "chunks_list": [],
                }
            }
        ),
        encoding="utf-8",
    )

    probe = inspect_provider_index("lightrag", version_dir)
    assert probe.ready is False
    assert "bad.docx" in probe.failure_summary

    (version_dir / "kv_store_doc_status.json").write_text(
        json.dumps(
            {
                "doc-1": {
                    "status": "processed",
                    "file_path": "ok.docx",
                    "chunks_list": ["chunk-1"],
                }
            }
        ),
        encoding="utf-8",
    )
    probe = inspect_provider_index("lightrag", version_dir)
    assert probe.ready is True
    assert probe.doc_count == 1


def test_provider_mismatch_is_not_ready(tmp_path: Path) -> None:
    version_dir = tmp_path / "version-1"
    version_dir.mkdir()
    _write_meta(version_dir, provider="lightrag")
    entry = {
        "provider": "lightrag",
        "signature": "lightrag",
        "ready": True,
        "storage_path": str(version_dir),
    }

    probe = inspect_provider_version(entry, "llamaindex")

    assert probe.ready is False
    assert probe.diagnostics["provider_mismatch"] is True
