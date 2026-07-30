"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Folder,
  FolderPlus,
  Loader2,
  MoveRight,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { invalidateClientCache } from "@/lib/client-cache";
import {
  createKbFolder,
  deleteKbFile,
  listKnowledgeBaseFiles,
  moveKbFile,
  renameKbFile,
  type KnowledgeBaseFile,
} from "@/lib/knowledge-api";
import { docIconFor, formatBytes } from "@/lib/doc-attachments";

interface KbDocumentListProps {
  kbName: string;
  /** Refresh trigger: bumping this prop forces a re-fetch (e.g. after upload). */
  refreshKey?: number;
  selectedFile: string | null;
  onSelect: (file: KnowledgeBaseFile | null) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

interface TreeNode {
  name: string; // segment label
  path: string; // full POSIX path relative to raw/
  type: "file" | "folder";
  file?: KnowledgeBaseFile;
  children: TreeNode[];
}

function parentOf(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx === -1 ? "" : path.slice(0, idx);
}

const DEFAULT_WIDTH = 220;
const MIN_WIDTH = 160;
const MAX_WIDTH = 560;
const WIDTH_STORAGE_KEY = "kb-document-list-width";

function buildTree(entries: KnowledgeBaseFile[]): {
  root: TreeNode[];
  folderPaths: string[];
} {
  const folders = new Map<string, TreeNode>();
  const root: TreeNode[] = [];
  const folderPaths: string[] = [];

  const ensureFolder = (path: string): TreeNode => {
    const existing = folders.get(path);
    if (existing) return existing;
    const node: TreeNode = {
      name: path.slice(path.lastIndexOf("/") + 1),
      path,
      type: "folder",
      children: [],
    };
    folders.set(path, node);
    folderPaths.push(path);
    const parent = parentOf(path);
    if (parent) ensureFolder(parent).children.push(node);
    else root.push(node);
    return node;
  };

  // Folders first so a file's parent always exists, and empty folders show.
  for (const entry of entries) {
    if (entry.type === "folder") ensureFolder(entry.name);
  }
  for (const entry of entries) {
    if (entry.type === "folder") continue;
    const node: TreeNode = {
      name: entry.name.slice(entry.name.lastIndexOf("/") + 1),
      path: entry.name,
      type: "file",
      file: entry,
      children: [],
    };
    const parent = parentOf(entry.name);
    if (parent) ensureFolder(parent).children.push(node);
    else root.push(node);
  }

  const sortNodes = (nodes: TreeNode[]) => {
    nodes.sort((a, b) => {
      if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
      return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
    });
    nodes.forEach((n) => n.children.length && sortNodes(n.children));
  };
  sortNodes(root);
  folderPaths.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  return { root, folderPaths };
}

export default function KbDocumentList({
  kbName,
  refreshKey = 0,
  selectedFile,
  onSelect,
  collapsed,
  onToggleCollapsed,
}: KbDocumentListProps) {
  const { t } = useTranslation();
  const [files, setFiles] = useState<KnowledgeBaseFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [moveMenuFor, setMoveMenuFor] = useState<string | null>(null);
  const [confirmBatchDelete, setConfirmBatchDelete] = useState(false);
  const [renameFor, setRenameFor] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [dragPath, setDragPath] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [batchMoveOpen, setBatchMoveOpen] = useState(false);
  const hasLoadedRef = useRef(false);
  const knownFoldersRef = useRef<Set<string>>(new Set());

  // Resizable panel width. SSR/hydration renders the default; the saved
  // width is applied after mount to avoid a server/client mismatch.
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  useEffect(() => {
    const saved = Number(window.localStorage.getItem(WIDTH_STORAGE_KEY));
    if (Number.isFinite(saved) && saved >= MIN_WIDTH && saved <= MAX_WIDTH) {
      setWidth(saved);
    }
  }, []);

  const startResize = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    // Capture the pointer on the handle itself: all subsequent move/up/
    // cancel events for this pointer are retargeted here — even when the
    // cursor leaves the window or a native drag starts over a draggable
    // file row. Listening on `window` instead could miss the release and
    // leave the panel following the mouse forever.
    const handle = e.currentTarget;
    handle.setPointerCapture(e.pointerId);
    const startX = e.clientX;
    const startWidth = width;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    const onMove = (ev: Event) => {
      const pointer = ev as PointerEvent;
      setWidth(
        Math.min(
          MAX_WIDTH,
          Math.max(MIN_WIDTH, startWidth + pointer.clientX - startX),
        ),
      );
    };
    const stop = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
      document.body.style.userSelect = previousUserSelect;
      setWidth((current) => {
        window.localStorage.setItem(WIDTH_STORAGE_KEY, String(current));
        return current;
      });
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  };

  const load = useCallback(
    async (force = false) => {
      setLoading(true);
      setError(null);
      try {
        if (force) invalidateClientCache(`knowledge:files:${kbName}`);
        const next = await listKnowledgeBaseFiles(kbName, { force });
        setFiles(next);
        const folderNames = next
          .filter((e) => e.type === "folder")
          .map((e) => e.name);
        setExpanded((prev) => {
          // First load: default-expand every folder so files are visible
          // without digging. Later reloads (move/delete/upload) preserve
          // the user's fold state; folders created since count as new and
          // open so a just-made folder never looks empty.
          if (!hasLoadedRef.current) return new Set(folderNames);
          const kept = new Set<string>();
          for (const name of folderNames) {
            if (prev.has(name) || !knownFoldersRef.current.has(name)) {
              kept.add(name);
            }
          }
          return kept;
        });
        knownFoldersRef.current = new Set(folderNames);
        hasLoadedRef.current = true;
        // Prune selections whose paths vanished (moved/deleted elsewhere).
        const existing = new Set(
          next.filter((e) => e.type !== "folder").map((e) => e.name),
        );
        setSelectedPaths((prev) => {
          const kept = [...prev].filter((p) => existing.has(p));
          return kept.length === prev.size ? prev : new Set(kept);
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    },
    [kbName],
  );

  useEffect(() => {
    void load(refreshKey > 0);
  }, [load, refreshKey]);

  const { root, folderPaths } = useMemo(() => buildTree(files), [files]);
  const fileEntries = useMemo(
    () => files.filter((e) => e.type !== "folder"),
    [files],
  );

  const toggleFolder = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const handleCreateFolder = async () => {
    const name = newFolderName.trim();
    if (!name) return;
    setBusy(true);
    try {
      await createKbFolder(kbName, name);
      setNewFolderName("");
      setNewFolderOpen(false);
      await load(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleMove = async (source: string, destFolder: string) => {
    setMoveMenuFor(null);
    setDropTarget(null);
    setDragPath(null);
    if (parentOf(source) === destFolder) return; // already there
    setBusy(true);
    try {
      await moveKbFile(kbName, source, destFolder);
      const basename = source.slice(source.lastIndexOf("/") + 1);
      const newPath = destFolder ? `${destFolder}/${basename}` : basename;
      await load(true);
      // Keep the preview pointed at the moved file if it was selected.
      if (selectedFile === source) {
        const moved = files.find((f) => f.name === source);
        onSelect({
          ...(moved ?? { name: newPath }),
          name: newPath,
          type: "file",
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleRename = async (source: string) => {
    const basename = source.slice(source.lastIndexOf("/") + 1);
    const newName = renameValue.trim();
    setRenameFor(null);
    if (!newName || newName === basename) return;
    setBusy(true);
    try {
      await renameKbFile(kbName, source, newName);
      // Keep the preview pointed at the renamed file if it was selected.
      if (selectedFile === source) {
        const parent = parentOf(source);
        const newPath = parent ? `${parent}/${newName}` : newName;
        const existing = files.find((f) => f.name === source);
        onSelect({
          ...(existing ?? { name: newPath }),
          name: newPath,
          type: "file",
        });
      }
      await load(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const toggleSelect = (path: string) =>
    setSelectedPaths((prev) => {
      setConfirmBatchDelete(false);
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const handleDeleteSelected = async () => {
    setConfirmBatchDelete(false);
    const targets = [...selectedPaths];
    if (!targets.length) return;
    setBusy(true);
    try {
      const results = await Promise.allSettled(
        targets.map((path) => deleteKbFile(kbName, path)),
      );
      const failed = targets.filter((_, i) => results[i].status === "rejected");
      const deleted = new Set(
        targets.filter((_, i) => results[i].status === "fulfilled"),
      );
      // Drop the preview if the file being shown was deleted.
      if (selectedFile && deleted.has(selectedFile)) onSelect(null);
      setSelectedPaths(new Set());
      await load(true);
      if (failed.length) {
        setError(
          t("Failed to delete {{count}} file(s)", { count: failed.length }),
        );
      }
    } finally {
      setBusy(false);
    }
  };

  const handleMoveSelected = async (destFolder: string) => {
    setBatchMoveOpen(false);
    const targets = [...selectedPaths].filter(
      (p) => parentOf(p) !== destFolder,
    );
    if (!targets.length) return;
    setBusy(true);
    try {
      // Moves are independent filesystem renames (no shared state on the
      // backend), so run them concurrently — sequential round-trips through
      // the dev proxy made a batch take ~0.3s per file.
      const results = await Promise.allSettled(
        targets.map((path) => moveKbFile(kbName, path, destFolder)),
      );
      const failed = targets.filter((_, i) => results[i].status === "rejected");
      setSelectedPaths(new Set());
      // One reload for the whole batch — not one per file — so the user's
      // fold state survives the move (and N files don't cost N refetches).
      await load(true);
      if (failed.length) {
        setError(
          t("Failed to move {{count}} file(s)", { count: failed.length }),
        );
      }
    } finally {
      setBusy(false);
    }
  };

  if (collapsed) {
    return (
      <aside className="flex h-full w-[44px] shrink-0 flex-col items-center gap-1 border-r border-[var(--border)] bg-[var(--card)]/40 py-2">
        <button
          type="button"
          onClick={onToggleCollapsed}
          title={t("Expand")}
          aria-label={t("Expand")}
          className="flex h-7 w-7 items-center justify-center rounded-md text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
        >
          <PanelLeftOpen size={13} strokeWidth={1.7} />
        </button>
        <div className="my-1 h-px w-6 bg-[var(--border)]/60" />
        <div className="flex w-full flex-1 flex-col items-center gap-0.5 overflow-y-auto pb-2">
          {fileEntries.map((file) => {
            const spec = docIconFor(file.name);
            const Icon = spec.Icon;
            const active = selectedFile === file.name;
            return (
              <button
                key={file.name}
                type="button"
                onClick={() => onSelect(file)}
                title={file.name}
                aria-label={file.name}
                className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-colors ${
                  active
                    ? "bg-[var(--primary)]/12 ring-1 ring-[var(--primary)]/40"
                    : "hover:bg-[var(--muted)]/60"
                }`}
              >
                {active && (
                  <span className="absolute -left-1 top-1/2 h-4 w-[2.5px] -translate-y-1/2 rounded-full bg-[var(--primary)]" />
                )}
                <Icon size={13} strokeWidth={1.6} className={spec.tint} />
              </button>
            );
          })}
        </div>
      </aside>
    );
  }

  const renderNode = (node: TreeNode, depth: number): React.ReactNode => {
    const indent = { paddingLeft: `${depth * 12 + 8}px` };
    if (node.type === "folder") {
      const open = expanded.has(node.path);
      const isDrop = dropTarget === node.path;
      return (
        <li key={`d:${node.path}`}>
          <div
            onClick={() => toggleFolder(node.path)}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDropTarget(node.path);
            }}
            onDragLeave={() =>
              setDropTarget((cur) => (cur === node.path ? null : cur))
            }
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              const src = e.dataTransfer.getData("text/plain");
              if (src) void handleMove(src, node.path);
            }}
            style={indent}
            className={`flex cursor-pointer items-center gap-1 rounded-md py-1.5 pr-2 text-left transition-colors ${
              isDrop
                ? "bg-[var(--primary)]/15 ring-1 ring-[var(--primary)]/40"
                : "hover:bg-[var(--muted)]/50"
            }`}
          >
            {open ? (
              <ChevronDown className="h-3 w-3 shrink-0 text-[var(--muted-foreground)]" />
            ) : (
              <ChevronRight className="h-3 w-3 shrink-0 text-[var(--muted-foreground)]" />
            )}
            <Folder className="h-3.5 w-3.5 shrink-0 text-[var(--muted-foreground)]" />
            <span className="truncate text-[12px] font-medium text-[var(--foreground)]">
              {node.name}
            </span>
          </div>
          {open && node.children.length > 0 && (
            <ul className="space-y-px">
              {node.children.map((child) => renderNode(child, depth + 1))}
            </ul>
          )}
        </li>
      );
    }

    const spec = docIconFor(node.name);
    const Icon = spec.Icon;
    const active = selectedFile === node.path;
    const file = node.file!;
    return (
      <li key={`f:${node.path}`} className="group/row relative">
        <div
          draggable
          onDragStart={(e) => {
            e.dataTransfer.setData("text/plain", node.path);
            e.dataTransfer.effectAllowed = "move";
            setDragPath(node.path);
          }}
          onDragEnd={() => setDragPath(null)}
          style={indent}
          className={`flex items-center gap-2 rounded-md py-1.5 pr-1 text-left transition-colors ${
            active
              ? "bg-[var(--primary)]/10 text-[var(--foreground)]"
              : selectedPaths.has(node.path)
                ? "bg-[var(--primary)]/6"
                : "hover:bg-[var(--muted)]/50"
          } ${dragPath === node.path ? "opacity-50" : ""}`}
        >
          <input
            type="checkbox"
            checked={selectedPaths.has(node.path)}
            onChange={() => toggleSelect(node.path)}
            onClick={(e) => e.stopPropagation()}
            title={t("Select file")}
            aria-label={t("Select file")}
            className="h-3 w-3 shrink-0 cursor-pointer accent-[var(--primary)]"
          />
          {renameFor === node.path ? (
            <input
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void handleRename(node.path);
                if (e.key === "Escape") setRenameFor(null);
              }}
              onBlur={() => void handleRename(node.path)}
              onClick={(e) => e.stopPropagation()}
              autoFocus
              disabled={busy}
              aria-label={t("Rename")}
              className="min-w-0 flex-1 rounded border border-[var(--primary)]/40 bg-[var(--background)] px-1.5 py-0.5 text-[12px] text-[var(--foreground)] outline-none"
            />
          ) : (
            <button
              type="button"
              onClick={() => onSelect(file)}
              title={node.path}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              <Icon
                size={13}
                strokeWidth={1.6}
                className={`shrink-0 ${spec.tint}`}
              />
              <div className="min-w-0 flex-1">
                <div className="truncate text-[12px] font-medium text-[var(--foreground)]">
                  {node.name}
                </div>
                <div className="truncate text-[10px] text-[var(--muted-foreground)]">
                  {file.size ? formatBytes(file.size) : ""}
                  {file.modified ? ` · ${formatRelative(file.modified)}` : ""}
                </div>
              </div>
            </button>
          )}
          <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover/row:opacity-100">
            <button
              type="button"
              onClick={() => {
                setMoveMenuFor(null);
                setRenameValue(node.name);
                setRenameFor(node.path);
              }}
              title={t("Rename")}
              aria-label={t("Rename")}
              className="rounded p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => {
                setRenameFor(null);
                setMoveMenuFor((cur) =>
                  cur === node.path ? null : node.path,
                );
              }}
              title={t("Move to…")}
              aria-label={t("Move to…")}
              className="rounded p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
            >
              <MoveRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {moveMenuFor === node.path && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={() => setMoveMenuFor(null)}
            />
            <div className="absolute right-1 top-8 z-20 max-h-60 w-44 overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--card)] py-1 shadow-lg">
              <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-[var(--muted-foreground)]">
                {t("Move to")}
              </div>
              {parentOf(node.path) !== "" && (
                <button
                  type="button"
                  onClick={() => void handleMove(node.path, "")}
                  className="block w-full truncate px-2.5 py-1.5 text-left text-[12px] text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]/60"
                >
                  / {t("Root")}
                </button>
              )}
              {folderPaths
                .filter((p) => p !== parentOf(node.path))
                .map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => void handleMove(node.path, p)}
                    className="block w-full truncate px-2.5 py-1.5 text-left text-[12px] text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]/60"
                  >
                    {p}
                  </button>
                ))}
              {folderPaths.length === 0 && parentOf(node.path) === "" && (
                <div className="px-2.5 py-1.5 text-[11px] text-[var(--muted-foreground)]">
                  {t("No folders yet")}
                </div>
              )}
            </div>
          </>
        )}
      </li>
    );
  };

  return (
    <aside
      style={{ width }}
      className="relative flex h-full shrink-0 flex-col border-r border-[var(--border)] bg-[var(--card)]/40"
    >
      <div className="flex items-center justify-between gap-1 px-2.5 pb-1.5 pt-2.5">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="text-[12px] font-medium text-[var(--foreground)]">
            {t("Files")}
          </span>
          <span className="rounded-full bg-[var(--muted)] px-1.5 py-0 text-[10px] text-[var(--muted-foreground)]">
            {fileEntries.length}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            onClick={() => {
              setNewFolderOpen((v) => !v);
              setNewFolderName("");
            }}
            title={t("New folder")}
            aria-label={t("New folder")}
            className="rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
          >
            <FolderPlus size={13} strokeWidth={1.7} />
          </button>
          <button
            type="button"
            onClick={() => void load(true)}
            title={t("Refresh")}
            aria-label={t("Refresh")}
            disabled={loading || busy}
            className="rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)] disabled:opacity-40"
          >
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
          </button>
          <button
            type="button"
            onClick={onToggleCollapsed}
            title={t("Collapse")}
            aria-label={t("Collapse")}
            className="rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
          >
            <PanelLeftClose size={12} strokeWidth={1.7} />
          </button>
        </div>
      </div>

      {newFolderOpen && (
        <div className="flex items-center gap-1 px-2.5 pb-1.5">
          <input
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleCreateFolder();
              if (e.key === "Escape") setNewFolderOpen(false);
            }}
            autoFocus
            placeholder={t("Folder name")}
            className="min-w-0 flex-1 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-[12px] text-[var(--foreground)] outline-none focus:border-[var(--foreground)]/25"
          />
          <button
            type="button"
            onClick={() => void handleCreateFolder()}
            disabled={busy || !newFolderName.trim()}
            className="shrink-0 rounded-md bg-[var(--primary)] px-2 py-1 text-[11px] font-medium text-[var(--primary-foreground)] transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {t("Add")}
          </button>
        </div>
      )}

      <div
        className="flex-1 overflow-y-auto px-1.5 pb-2.5"
        onDragOver={(e) => {
          e.preventDefault();
          setDropTarget("");
        }}
        onDrop={(e) => {
          e.preventDefault();
          const src = e.dataTransfer.getData("text/plain");
          if (src) void handleMove(src, "");
        }}
      >
        {error ? (
          <div className="rounded-md border border-red-200 bg-red-50 px-2.5 py-2 text-[11px] text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">
            {error}
            <button
              type="button"
              onClick={() => void load(true)}
              className="ml-1 underline"
            >
              {t("Retry")}
            </button>
          </div>
        ) : loading && !files.length ? (
          <div className="space-y-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-8 rounded-md bg-[var(--muted)]/40 animate-pulse"
              />
            ))}
          </div>
        ) : fileEntries.length === 0 && folderPaths.length === 0 ? (
          <div className="px-2 py-6 text-center text-[11px] text-[var(--muted-foreground)]">
            <Folder className="mx-auto mb-1.5 h-3.5 w-3.5 opacity-50" />
            {t("No files yet. Add one using the Add Documents tab.")}
          </div>
        ) : (
          <ul className="space-y-px">{root.map((n) => renderNode(n, 0))}</ul>
        )}
      </div>

      {selectedPaths.size > 0 && (
        <div className="relative flex shrink-0 items-center gap-1.5 border-t border-[var(--border)] px-2.5 py-2">
          <span className="min-w-0 flex-1 truncate text-[11px] text-[var(--muted-foreground)]">
            {t("{{count}} selected", { count: selectedPaths.size })}
          </span>
          <button
            type="button"
            onClick={() => setBatchMoveOpen((v) => !v)}
            disabled={busy}
            className="flex shrink-0 items-center gap-1 rounded-md bg-[var(--primary)] px-2 py-1 text-[11px] font-medium text-[var(--primary-foreground)] transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            <MoveRight size={11} strokeWidth={1.8} />
            {t("Move to…")}
          </button>
          {confirmBatchDelete ? (
            <>
              <button
                type="button"
                onClick={() => void handleDeleteSelected()}
                disabled={busy}
                title={t("Confirm delete")}
                aria-label={t("Confirm delete")}
                className="flex shrink-0 items-center gap-0.5 rounded-md bg-red-600 px-2 py-1 text-[11px] font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                <Check size={11} strokeWidth={2} />
                {t("Delete?")}
              </button>
              <button
                type="button"
                onClick={() => setConfirmBatchDelete(false)}
                title={t("Cancel")}
                aria-label={t("Cancel")}
                className="shrink-0 rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
              >
                <X size={12} strokeWidth={1.8} />
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => {
                setBatchMoveOpen(false);
                setConfirmBatchDelete(true);
              }}
              disabled={busy}
              title={t("Delete selected")}
              aria-label={t("Delete selected")}
              className="shrink-0 rounded-md p-1.5 text-[var(--muted-foreground)] transition-colors hover:bg-red-50 hover:text-red-600 disabled:opacity-40 dark:hover:bg-red-950/40 dark:hover:text-red-400"
            >
              <Trash2 size={13} strokeWidth={1.7} />
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              setSelectedPaths(new Set());
              setConfirmBatchDelete(false);
            }}
            title={t("Clear selection")}
            aria-label={t("Clear selection")}
            className="shrink-0 rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] hover:text-[var(--foreground)]"
          >
            <X size={12} strokeWidth={1.8} />
          </button>
          {batchMoveOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setBatchMoveOpen(false)}
              />
              <div className="absolute bottom-9 right-2 z-20 max-h-60 w-44 overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--card)] py-1 shadow-lg">
                <div className="px-2.5 py-1 text-[10px] uppercase tracking-wide text-[var(--muted-foreground)]">
                  {t("Move to")}
                </div>
                <button
                  type="button"
                  onClick={() => void handleMoveSelected("")}
                  className="block w-full truncate px-2.5 py-1.5 text-left text-[12px] text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]/60"
                >
                  / {t("Root")}
                </button>
                {folderPaths.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => void handleMoveSelected(p)}
                    className="block w-full truncate px-2.5 py-1.5 text-left text-[12px] text-[var(--foreground)] transition-colors hover:bg-[var(--muted)]/60"
                  >
                    {p}
                  </button>
                ))}
                {folderPaths.length === 0 && (
                  <div className="px-2.5 py-1.5 text-[11px] text-[var(--muted-foreground)]">
                    {t("No folders yet")}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* Resize handle on the divider; double-click snaps back to default. */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label={t("Drag to resize")}
        title={t("Drag to resize")}
        onPointerDown={startResize}
        onDoubleClick={() => {
          setWidth(DEFAULT_WIDTH);
          window.localStorage.setItem(WIDTH_STORAGE_KEY, String(DEFAULT_WIDTH));
        }}
        className="absolute inset-y-0 right-0 z-10 w-1.5 cursor-col-resize transition-colors hover:bg-[var(--primary)]/30 active:bg-[var(--primary)]/50"
      />
    </aside>
  );
}

function formatRelative(unixSeconds: number): string {
  const ts = unixSeconds * 1000;
  const diff = Date.now() - ts;
  if (diff < 60_000) return "just now";
  if (diff < 60 * 60_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 24 * 60 * 60_000)
    return `${Math.floor(diff / (60 * 60_000))}h ago`;
  if (diff < 30 * 24 * 60 * 60_000)
    return `${Math.floor(diff / (24 * 60 * 60_000))}d ago`;
  return new Date(ts).toLocaleDateString();
}
