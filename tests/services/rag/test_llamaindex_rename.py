"""Tests for llamaindex source-reference rewriting after a raw file rename."""

from __future__ import annotations

import json
from pathlib import Path

from deeptutor.services.rag.pipelines.llamaindex.storage import rename_source_references


def _write_index(storage_dir: Path, raw_dir: Path) -> None:
    """Minimal llamaindex storage: docstore + BM25 corpus for two raw files.

    ``a.txt`` lives at the raw root; ``folder/a.txt`` shares the basename so
    tests can prove retitling only touches exact ``file_path`` matches.
    """
    storage_dir.mkdir(parents=True)
    (storage_dir / "bm25_retriever").mkdir()

    def meta(rel: str) -> dict:
        return {"file_name": Path(rel).name, "file_path": str(raw_dir / rel)}

    docstore = {
        "docstore/data": {
            "node-1": {"__data__": {"metadata": meta("a.txt")}},
            "node-2": {"__data__": {"metadata": meta("folder/a.txt")}},
        },
        "docstore/ref_doc_info": {
            "doc-1": {"node_ids": ["node-1"], "metadata": meta("a.txt")},
            "doc-2": {"node_ids": ["node-2"], "metadata": meta("folder/a.txt")},
        },
        "docstore/metadata": {},
    }
    (storage_dir / "docstore.json").write_text(
        json.dumps(docstore, ensure_ascii=False), encoding="utf-8"
    )
    (storage_dir / "index_store.json").write_text("{}", encoding="utf-8")

    rows = []
    for rel, node_id in (("a.txt", "node-1"), ("folder/a.txt", "node-2")):
        m = meta(rel)
        rows.append(
            json.dumps(
                {
                    **m,
                    "_node_content": json.dumps(
                        {"id_": node_id, "metadata": m}, ensure_ascii=False
                    ),
                },
                ensure_ascii=False,
            )
        )
    (storage_dir / "bm25_retriever" / "corpus.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def test_rename_patches_docstore_and_bm25(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    storage_dir = tmp_path / "version-1"
    _write_index(storage_dir, raw_dir)

    patched = rename_source_references(
        storage_dir, old_path=raw_dir / "a.txt", new_path=raw_dir / "b.txt"
    )

    assert patched > 0
    docstore = json.loads((storage_dir / "docstore.json").read_text(encoding="utf-8"))
    renamed = docstore["docstore/data"]["node-1"]["__data__"]["metadata"]
    assert renamed["file_name"] == "b.txt"
    assert renamed["file_path"] == str(raw_dir / "b.txt")
    ref = docstore["docstore/ref_doc_info"]["doc-1"]["metadata"]
    assert ref["file_name"] == "b.txt"
    assert ref["file_path"] == str(raw_dir / "b.txt")

    rows = [
        json.loads(line)
        for line in (storage_dir / "bm25_retriever" / "corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["file_name"] == "b.txt"
    assert rows[0]["file_path"] == str(raw_dir / "b.txt")
    node = json.loads(rows[0]["_node_content"])
    assert node["metadata"]["file_name"] == "b.txt"
    assert node["metadata"]["file_path"] == str(raw_dir / "b.txt")


def test_rename_leaves_same_basename_in_other_folder_untouched(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    storage_dir = tmp_path / "version-1"
    _write_index(storage_dir, raw_dir)

    rename_source_references(storage_dir, old_path=raw_dir / "a.txt", new_path=raw_dir / "b.txt")

    docstore = json.loads((storage_dir / "docstore.json").read_text(encoding="utf-8"))
    other = docstore["docstore/data"]["node-2"]["__data__"]["metadata"]
    assert other["file_name"] == "a.txt"
    assert other["file_path"] == str(raw_dir / "folder" / "a.txt")

    rows = [
        json.loads(line)
        for line in (storage_dir / "bm25_retriever" / "corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[1]["file_name"] == "a.txt"
    node = json.loads(rows[1]["_node_content"])
    assert node["metadata"]["file_name"] == "a.txt"


def test_rename_without_index_is_a_noop(tmp_path: Path) -> None:
    storage_dir = tmp_path / "version-1"
    storage_dir.mkdir()
    patched = rename_source_references(
        storage_dir, old_path=tmp_path / "a.txt", new_path=tmp_path / "b.txt"
    )
    assert patched == 0
