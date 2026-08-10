"""Migrate a LightRAG KB version's vector stores from NanoVectorDB to Faiss.

NanoVectorDB keeps every vector in a JSON file that is fully decoded into RAM
on open (vdb_chunks/vdb_entities/vdb_relationships). Faiss keeps a compact
binary index on disk that is cheap to load. This script converts in place:

    vdb_<ns>.json            ->  faiss_index_<ns>.index (+ .index.meta.json)

No LLM / embedding calls are needed: the source JSONs already contain every
vector (the base64 ``matrix`` is row-aligned with ``data``). After conversion
the version's ``meta.json`` is stamped ``vector_storage: "faiss"`` so the
pipeline opens it with Faiss from then on. The nano JSON files are LEFT in
place (delete them manually once you have verified the migration; they are no
longer read).

Usage:
    python scripts/migrate_lightrag_vector_storage.py <version_dir> [--force] [--no-backup]

Example:
    python scripts/migrate_lightrag_vector_storage.py data/knowledge_bases/pmpp-5th/version-1
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

# Namespaces and meta-field whitelists mirror LightRAG's own construction of
# chunks_vdb / entities_vdb / relationships_vdb (lightrag.lightrag.LightRAG).
NAMESPACES = ("chunks", "entities", "relationships")
META_FIELDS = {
    "chunks": {"full_doc_id", "content", "file_path"},
    "entities": {"entity_name", "source_id", "content", "file_path"},
    "relationships": {"src_id", "tgt_id", "source_id", "content", "file_path"},
}


def _clone_backup(src: Path) -> Path:
    """APFS copy-on-write clone (instant, ~no extra disk); plain copy fallback."""
    backup = src.with_name(src.name + ".faiss-migration-bak")
    if backup.exists():
        raise SystemExit(f"Backup path already exists: {backup} — remove it or investigate first.")
    result = subprocess.run(["cp", "-Rc", str(src), str(backup)], capture_output=True)
    if result.returncode != 0:
        import shutil

        shutil.copytree(src, backup)
    return backup


def _convert_namespace(version_dir: Path, ns: str, *, force: bool) -> tuple[int, int]:
    """Convert one vdb_<ns>.json. Returns (records, dim). Skips absent files."""
    import faiss

    nano_file = version_dir / f"vdb_{ns}.json"
    if not nano_file.exists():
        print(f"[{ns}] no vdb_{ns}.json — skipped")
        return (0, 0)

    index_file = version_dir / f"faiss_index_{ns}.index"
    meta_file = version_dir / f"faiss_index_{ns}.index.meta.json"
    if (index_file.exists() or meta_file.exists()) and not force:
        raise SystemExit(f"[{ns}] Faiss files already exist — pass --force to overwrite.")

    print(f"[{ns}] loading {nano_file.name} ({nano_file.stat().st_size / 1e6:.0f} MB)…")
    with open(nano_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    data = payload["data"]
    dim = int(payload["embedding_dim"])
    matrix = np.frombuffer(base64.b64decode(payload["matrix"]), dtype=np.float32)
    del payload
    if matrix.size != len(data) * dim:
        raise SystemExit(
            f"[{ns}] matrix/data mismatch: {matrix.size} floats for {len(data)} records x {dim}d"
        )
    matrix = matrix.reshape(len(data), dim)
    print(f"[{ns}] {len(data)} records x {dim}d")

    # FaissVectorDBStorage uses IndexFlatIP over L2-normalized vectors (cosine).
    vectors = matrix.copy()
    del matrix
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    del vectors
    gc.collect()

    tmp_index = index_file.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_index))
    os.replace(tmp_index, index_file)

    meta = {}
    for fid, record in enumerate(data):
        entry = {f: record[f] for f in META_FIELDS[ns] if f in record}
        entry["__id__"] = record["__id__"]
        entry["__created_at__"] = int(record.get("__created_at__") or 0)
        meta[str(fid)] = entry
    tmp_meta = meta_file.with_suffix(".tmp")
    with open(tmp_meta, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)
    os.replace(tmp_meta, meta_file)
    del meta, data, index
    gc.collect()

    # Verify: reload and check counts, then probe with an existing vector —
    # its own row must come back as the top hit with similarity ~1.0.
    reloaded = faiss.read_index(str(index_file))
    if reloaded.ntotal == 0:
        raise SystemExit(f"[{ns}] reloaded index is empty")
    row = np.array([reloaded.reconstruct(0)], dtype=np.float32)
    distances, indices = reloaded.search(row, 1)
    if int(indices[0][0]) != 0 or distances[0][0] < 0.999:
        raise SystemExit(
            f"[{ns}] self-probe failed (top1={indices[0][0]}, sim={distances[0][0]:.4f})"
        )
    with open(meta_file, encoding="utf-8") as handle:
        written = json.load(handle)
    if len(written) != reloaded.ntotal:
        raise SystemExit(f"[{ns}] meta rows {len(written)} != index rows {reloaded.ntotal}")
    print(f"[{ns}] verified: {reloaded.ntotal} vectors, self-probe sim {distances[0][0]:.4f}")
    return (reloaded.ntotal, dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_dir", type=Path, help="KB version dir (LightRAG working_dir)")
    parser.add_argument("--force", action="store_true", help="overwrite existing Faiss files")
    parser.add_argument("--no-backup", action="store_true", help="skip the safety clone")
    args = parser.parse_args()

    version_dir = args.version_dir.resolve()
    if not version_dir.is_dir():
        raise SystemExit(f"Not a directory: {version_dir}")
    meta_path = version_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"No meta.json in {version_dir} — is this a LightRAG version dir?")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("vector_storage") == "faiss" and not args.force:
        raise SystemExit("meta.json already says faiss — nothing to do (use --force to redo).")

    if not args.no_backup:
        backup = _clone_backup(version_dir)
        print(f"backup clone: {backup}")

    totals = {}
    for ns in NAMESPACES:
        count, dim = _convert_namespace(version_dir, ns, force=args.force)
        if count:
            totals[ns] = (count, dim)

    if not totals:
        raise SystemExit("Nothing converted — no vdb_*.json files found.")

    dims = {d for _, d in totals.values()}
    if len(dims) != 1:
        raise SystemExit(f"Inconsistent embedding dims across namespaces: {totals}")

    meta["vector_storage"] = "faiss"
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta_path)

    print("\nDone. Converted:", {k: v[0] for k, v in totals.items()})
    print(f"meta.json: vector_storage = faiss ({meta_path})")
    print("The old vdb_*.json files were left untouched; delete them after verifying queries.")


if __name__ == "__main__":
    sys.exit(main())
