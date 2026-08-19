"""Publish a pre-built knowledge base from the administrator upload area.

This is deliberately not an indexing path.  An administrator places a complete
KB directory below ``data/upload`` and this module validates, copies, and
registers it below ``data/knowledge_bases`` without invoking an LLM, embedding
provider, parser, or RAG build pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any
from uuid import uuid4

from deeptutor.knowledge.manager import KnowledgeBaseManager
from deeptutor.knowledge.naming import validate_knowledge_base_name
from deeptutor.knowledge.policy import is_kb_query_only
from deeptutor.services.file_io import atomic_write_json
from deeptutor.services.path_service import PathService
from deeptutor.services.rag.factory import (
    DEFAULT_PROVIDER,
    GRAPHRAG_PROVIDER,
    LIGHTRAG_PROVIDER,
    normalize_provider_name,
)
from deeptutor.services.rag.index_probe import inspect_kb_versions
from deeptutor.services.rag.index_versioning import list_kb_versions
from deeptutor.services.rag.linked_kb import probe_linked_folder

IMPORTABLE_PROVIDERS = frozenset({DEFAULT_PROVIDER, GRAPHRAG_PROVIDER, LIGHTRAG_PROVIDER})
_LIVE_PROGRESS_STAGES = frozenset(
    {"initializing", "processing_documents", "processing_file", "processing"}
)


@dataclass(frozen=True)
class UploadFolder:
    name: str
    path: str
    candidate: bool


@dataclass(frozen=True)
class UploadFolderListing:
    path: str
    parent: str | None
    candidate: bool
    folders: list[UploadFolder]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["folders"] = [asdict(folder) for folder in self.folders]
        return payload


@dataclass(frozen=True)
class ImportProbe:
    ok: bool
    path: str
    suggested_name: str
    provider: str
    version_count: int
    ready_version_count: int
    document_count: int | None
    file_count: int
    size_bytes: int
    warnings: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def upload_root_for(manager: KnowledgeBaseManager) -> Path:
    """Return the fixed upload root beside this manager's KB directory."""
    return PathService(workspace_root=manager.base_dir.parent).get_knowledge_base_upload_root()


def _normalize_relative_path(relative_path: str) -> str:
    raw = str(relative_path or "").strip().replace("\\", "/")
    path = PurePosixPath(raw or ".")
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        if raw in {"", "."}:
            return ""
        raise ValueError("Upload folder path must stay inside the upload directory.")
    return "" if raw in {"", "."} else path.as_posix()


def resolve_upload_folder(manager: KnowledgeBaseManager, relative_path: str) -> tuple[Path, str]:
    """Resolve a client-supplied relative path beneath ``data/upload``."""
    normalized = _normalize_relative_path(relative_path)
    root = upload_root_for(manager).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Upload folder path must stay inside the upload directory.") from exc
    if not candidate.is_dir():
        raise ValueError("Upload folder does not exist or is not a directory.")
    return candidate, normalized


def _looks_like_kb(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    if (folder / "metadata.json").is_file():
        return True
    try:
        return any(
            child.is_dir() and child.name.startswith("version-") for child in folder.iterdir()
        )
    except OSError:
        return False


def list_upload_folders(
    manager: KnowledgeBaseManager, relative_path: str = ""
) -> UploadFolderListing:
    folder, normalized = resolve_upload_folder(manager, relative_path)
    children: list[UploadFolder] = []
    for child in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        try:
            if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
                continue
            resolved = child.resolve()
            resolved.relative_to(upload_root_for(manager).resolve())
        except (OSError, ValueError):
            continue
        child_relative = resolved.relative_to(upload_root_for(manager).resolve()).as_posix()
        children.append(
            UploadFolder(
                name=child.name,
                path=child_relative,
                candidate=_looks_like_kb(child),
            )
        )
    parent = None
    if normalized:
        parent_path = PurePosixPath(normalized).parent
        parent = "" if str(parent_path) == "." else parent_path.as_posix()
    return UploadFolderListing(
        path=normalized,
        parent=parent,
        candidate=_looks_like_kb(folder),
        folders=children,
    )


def _read_json_object(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ValueError(f"Missing required file: {path.name}")
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path.name}")
    return payload


def _detect_provider(folder: Path, metadata: dict[str, Any]) -> str:
    raw_provider = str(metadata.get("rag_provider") or "").strip().lower()
    if raw_provider:
        provider = normalize_provider_name(raw_provider)
        if provider != raw_provider:
            raise ValueError(f"Unsupported RAG provider in metadata.json: {raw_provider}")
        return provider
    versions = list_kb_versions(folder)
    for version in versions:
        provider = str(version.get("provider") or version.get("signature") or "").lower()
        if provider in IMPORTABLE_PROVIDERS:
            return provider
    return DEFAULT_PROVIDER


def _validate_regular_tree(folder: Path) -> tuple[int, int]:
    """Reject links and special files; return ``(file_count, size_bytes)``."""
    file_count = 0
    size_bytes = 0
    for current, dir_names, file_names in os.walk(folder, followlinks=False):
        current_path = Path(current)
        for name in [*dir_names, *file_names]:
            path = current_path / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ValueError(f"Cannot inspect upload entry: {path.name}") from exc
            if stat.S_ISLNK(mode):
                raise ValueError("Knowledge base uploads cannot contain symbolic links.")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("Knowledge base uploads can contain only regular files.")
            file_count += 1
            size_bytes += path.stat().st_size
    return file_count, size_bytes


def _probe_directory(folder: Path, relative_path: str) -> ImportProbe:
    metadata: dict[str, Any] = {}
    provider = DEFAULT_PROVIDER
    suggested_name = folder.name
    file_count = 0
    size_bytes = 0
    warnings: list[str] = []
    try:
        # Validate the whole tree before opening even metadata/progress files.
        # Otherwise a crafted upload could make those files symbolic links and
        # cause the probe to read outside the fixed upload directory before the
        # later copy-time validation rejected it.
        file_count, size_bytes = _validate_regular_tree(folder)
        metadata = _read_json_object(folder / "metadata.json", required=True)
        suggested_name = validate_knowledge_base_name(str(metadata.get("name") or folder.name))
        if not (folder / "raw").is_dir():
            raise ValueError("Missing required raw directory.")
        progress = _read_json_object(folder / ".progress.json")
        progress_stage = str(progress.get("stage") or "").strip().lower()
        if progress_stage in _LIVE_PROGRESS_STAGES:
            raise ValueError("Knowledge base is still being built; wait for it to complete.")
        if progress_stage == "error":
            raise ValueError("Knowledge base build ended with an error.")

        provider = _detect_provider(folder, metadata)
        if provider not in IMPORTABLE_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' is not a portable local knowledge-base engine."
            )

        versions = inspect_kb_versions(folder, provider)
        if not versions:
            raise ValueError("No knowledge-base index versions were found.")
        incomplete = [
            str(item.get("version") or "unknown") for item in versions if not item.get("ready")
        ]
        if incomplete:
            raise ValueError("Incomplete or incompatible index versions: " + ", ".join(incomplete))
        ready = [item for item in versions if item.get("ready")]
        if not ready:
            raise ValueError("No ready index version was found for this knowledge base.")
        compatibility = probe_linked_folder(str(folder), provider)
        if not compatibility.ok:
            raise ValueError(compatibility.error or "The index is not queryable.")
        warnings.extend(compatibility.warnings)
        doc_count = compatibility.doc_count
        if not progress:
            warnings.append(
                "No .progress.json snapshot was found; index files were validated directly."
            )
        return ImportProbe(
            ok=True,
            path=relative_path,
            suggested_name=suggested_name,
            provider=provider,
            version_count=len(versions),
            ready_version_count=len(ready),
            document_count=doc_count,
            file_count=file_count,
            size_bytes=size_bytes,
            warnings=warnings,
        )
    except ValueError as exc:
        return ImportProbe(
            ok=False,
            path=relative_path,
            suggested_name=suggested_name,
            provider=provider,
            version_count=0,
            ready_version_count=0,
            document_count=None,
            file_count=file_count,
            size_bytes=size_bytes,
            warnings=warnings,
            error=str(exc),
        )


def probe_import_folder(manager: KnowledgeBaseManager, relative_path: str) -> ImportProbe:
    folder, normalized = resolve_upload_folder(manager, relative_path)
    if not normalized:
        raise ValueError("Select a knowledge-base folder inside the upload directory.")
    return _probe_directory(folder, normalized)


def _build_config_entry(
    target: Path,
    name: str,
    provider: str,
    relative_path: str,
) -> dict[str, Any]:
    metadata = _read_json_object(target / "metadata.json", required=True)
    versions = inspect_kb_versions(target, provider)
    latest = next(item for item in versions if item.get("ready"))
    now = datetime.now().isoformat()
    entry: dict[str, Any] = {
        "path": name,
        "description": str(metadata.get("description") or f"Knowledge base: {name}"),
        "status": "ready",
        "rag_provider": provider,
        "needs_reindex": False,
        "created_at": metadata.get("created_at") or now,
        "updated_at": now,
        "imported_at": now,
        "imported_from": relative_path,
        "index_versions": versions,
    }
    for key in ("last_indexed_at", "last_indexed_count", "last_indexed_action"):
        if metadata.get(key) is not None:
            entry[key] = metadata[key]
    model = latest.get("embedding_model") or latest.get("model")
    dimension = latest.get("embedding_dim") or latest.get("dimension")
    signature = latest.get("embedding_signature") or latest.get("signature")
    if model:
        entry["embedding_model"] = model
    if dimension is not None:
        entry["embedding_dim"] = dimension
    if signature:
        entry["embedding_signature"] = signature
    return entry


def _rewrite_copied_identity(target: Path, name: str) -> None:
    for filename in ("metadata.json", ".progress.json"):
        path = target / filename
        if not path.exists():
            continue
        payload = _read_json_object(path, required=True)
        key = "name" if filename == "metadata.json" else "kb_name"
        payload[key] = name
        atomic_write_json(path, payload)


def _relocate_version_paths(entry: dict[str, Any], source: Path, target: Path) -> None:
    """Rewrite staged absolute paths to their final published location."""
    for version in entry.get("index_versions", []):
        if not isinstance(version, dict):
            continue
        for key in ("storage_path", "version_path", "kb_path"):
            value = version.get(key)
            if not isinstance(value, str):
                continue
            try:
                relative = Path(value).relative_to(source)
            except ValueError:
                continue
            version[key] = str(target / relative)


def import_existing_knowledge_base(
    manager: KnowledgeBaseManager,
    relative_path: str,
    requested_name: str = "",
) -> dict[str, Any]:
    """Copy and atomically register one validated KB from ``data/upload``."""
    source, normalized = resolve_upload_folder(manager, relative_path)
    if not normalized:
        raise ValueError("Select a knowledge-base folder inside the upload directory.")
    probe = _probe_directory(source, normalized)
    if not probe.ok:
        raise ValueError(probe.error or "Knowledge base upload is not importable.")
    name = validate_knowledge_base_name(requested_name or probe.suggested_name)

    manager.base_dir.mkdir(parents=True, exist_ok=True)
    target = manager.base_dir / name
    if target.exists():
        raise ValueError(f"A knowledge base named '{name}' already exists.")

    free_bytes = shutil.disk_usage(manager.base_dir).free
    if free_bytes < probe.size_bytes:
        raise ValueError("Not enough free disk space to import this knowledge base.")

    staged = manager.base_dir / f".importing-{uuid4().hex}"
    published = False
    try:
        shutil.copytree(source, staged, copy_function=shutil.copy2)
        copied_probe = _probe_directory(staged, normalized)
        if not copied_probe.ok:
            raise ValueError(
                "Copied knowledge base failed validation: "
                + (copied_probe.error or "unknown error")
            )
        _rewrite_copied_identity(staged, name)
        config_entry = _build_config_entry(staged, name, probe.provider, normalized)
        _relocate_version_paths(config_entry, staged, target)

        transaction = manager._transact_policy_exception if is_kb_query_only() else manager.transact
        with transaction() as config:
            knowledge_bases = config.setdefault("knowledge_bases", {})
            if name in knowledge_bases or target.exists():
                raise ValueError(f"A knowledge base named '{name}' already exists.")
            os.replace(staged, target)
            published = True
            knowledge_bases[name] = config_entry
        manager._sync_kb_to_pb(name, config_entry)
        return {
            "status": "imported",
            "name": name,
            "source_path": normalized,
            "rag_provider": probe.provider,
            "file_count": copied_probe.file_count,
            "size_bytes": copied_probe.size_bytes,
        }
    except Exception:
        if published and target.exists():
            try:
                os.replace(target, staged)
                published = False
            except OSError:
                pass
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)


__all__ = [
    "ImportProbe",
    "UploadFolderListing",
    "import_existing_knowledge_base",
    "list_upload_folders",
    "probe_import_folder",
    "resolve_upload_folder",
    "upload_root_for",
]
