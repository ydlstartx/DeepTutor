"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Check,
  ChevronDown,
  Folder,
  FolderPlus,
  History,
  Loader2,
  Pencil,
  RefreshCw,
  Search,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import OrganizedSessionList from "@/components/courses/OrganizedSessionList";
import SpaceSectionHeader from "@/components/space/SpaceSectionHeader";
import { useAppShell } from "@/context/AppShellContext";
import {
  createSessionFolder,
  deleteSession,
  deleteSessionFolder,
  listAllSessions,
  listSessionFolders,
  moveSessionToFolder,
  renameSessionFolder,
  updateSessionTitle,
  updateSessionOrganization,
  type SessionFolder,
  type SessionOrganizationPatch,
  type SessionSummary,
} from "@/lib/session-api";
import { listCourses, type StudyCourse } from "@/lib/courses-api";

/** Sessions list for chat history. Reopened sessions route to the main chat. */
export interface ChatHistorySectionProps {
  icon?: LucideIcon;
  title?: string;
  description?: string;
}

interface FolderGroup {
  id: string | null;
  name: string;
  sessions: SessionSummary[];
}

export default function ChatHistorySection({
  icon,
  title,
  description,
}: ChatHistorySectionProps = {}) {
  const basePath = "/home";
  const { t } = useTranslation();
  const router = useRouter();
  const { activeSessionId, setActiveSessionId } = useAppShell();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [folders, setFolders] = useState<SessionFolder[]>([]);
  const [courses, setCourses] = useState<StudyCourse[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [courseFilter, setCourseFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [archiveFilter, setArchiveFilter] = useState("active");
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const [folderDraft, setFolderDraft] = useState("");
  const [folderError, setFolderError] = useState<string | null>(null);
  const [collapsedFolderIds, setCollapsedFolderIds] = useState<Set<string>>(
    () => new Set(),
  );

  const describeError = useCallback(
    (error: unknown, fallbackKey: string) =>
      error instanceof Error ? t(error.message) : t(fallbackKey),
    [t],
  );

  const load = useCallback(async (force = false) => {
    setLoading(true);
    setFolderError(null);
    try {
      const [nextSessions, nextFolders, nextCourses] = await Promise.all([
        listAllSessions({ force }),
        listSessionFolders({ force }),
        listCourses({ force }),
      ]);
      setSessions(nextSessions);
      setFolders(nextFolders);
      setCourses(nextCourses);
    } catch (error) {
      setFolderError(describeError(error, "Failed to load chat history"));
    } finally {
      setLoading(false);
    }
  }, [describeError]);

  useEffect(() => {
    void load(true);
  }, [load]);

  const filteredSessions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return sessions.filter((session) => {
      const prefs = session.preferences ?? {};
      if (archiveFilter === "active" && prefs.archived) return false;
      if (archiveFilter === "archived" && !prefs.archived) return false;
      if (courseFilter === "unclassified" && prefs.course_id) return false;
      if (
        courseFilter !== "all" &&
        courseFilter !== "unclassified" &&
        prefs.course_id !== courseFilter
      )
        return false;
      if (kindFilter === "chat" && prefs.session_kind === "selection_tutor")
        return false;
      if (
        kindFilter === "selection_tutor" &&
        prefs.session_kind !== "selection_tutor"
      )
        return false;
      if (!needle) return true;
      return [session.title, session.last_message]
        .filter(Boolean)
        .some((value) => value.toLowerCase().includes(needle));
    });
  }, [archiveFilter, courseFilter, kindFilter, query, sessions]);

  const groups = useMemo<FolderGroup[]>(() => {
    const byFolder = new Map<string, SessionSummary[]>();
    const uncategorized: SessionSummary[] = [];
    const knownFolderIds = new Set(folders.map((folder) => folder.id));
    for (const session of filteredSessions) {
      if (!session.folder_id || !knownFolderIds.has(session.folder_id)) {
        uncategorized.push(session);
        continue;
      }
      const bucket = byFolder.get(session.folder_id) ?? [];
      bucket.push(session);
      byFolder.set(session.folder_id, bucket);
    }
    return [
      ...folders.map((folder) => ({
        id: folder.id,
        name: folder.name,
        sessions: byFolder.get(folder.id) ?? [],
      })),
      {
        id: null,
        name: t("Uncategorized"),
        sessions: uncategorized,
      },
    ];
  }, [filteredSessions, folders, t]);

  const handleSelect = useCallback(
    (sessionId: string) => {
      setActiveSessionId(sessionId);
      router.push(`${basePath}/${sessionId}`);
    },
    [basePath, router, setActiveSessionId],
  );

  const handleRename = useCallback(
    async (sessionId: string, nextTitle: string) => {
      await updateSessionTitle(sessionId, nextTitle);
      await load(true);
    },
    [load],
  );

  const handleDelete = useCallback(
    async (sessionId: string) => {
      if (!window.confirm(t("Delete this chat?"))) return;
      await deleteSession(sessionId);
      if (activeSessionId === sessionId) setActiveSessionId(null);
      await load(true);
    },
    [activeSessionId, load, setActiveSessionId, t],
  );

  const handleCreateFolder = async () => {
    const name = newFolderName.trim();
    if (!name) return;
    setFolderError(null);
    try {
      await createSessionFolder(name);
      setNewFolderName("");
      setCreatingFolder(false);
      await load(true);
    } catch (error) {
      setFolderError(describeError(error, "Could not create folder"));
    }
  };

  const handleRenameFolder = async (folderId: string) => {
    const name = folderDraft.trim();
    if (!name) return;
    setFolderError(null);
    try {
      await renameSessionFolder(folderId, name);
      setEditingFolderId(null);
      setFolderDraft("");
      await load(true);
    } catch (error) {
      setFolderError(describeError(error, "Could not rename folder"));
    }
  };

  const handleDeleteFolder = async (folder: SessionFolder) => {
    if (
      !window.confirm(
        t(
          'Delete folder "{{name}}"? Its conversations will move to Uncategorized.',
          { name: folder.name },
        ),
      )
    )
      return;
    setFolderError(null);
    try {
      await deleteSessionFolder(folder.id);
      await load(true);
    } catch (error) {
      setFolderError(describeError(error, "Could not delete folder"));
    }
  };

  const handleMove = async (sessionId: string, folderId: string | null) => {
    setFolderError(null);
    try {
      await moveSessionToFolder(sessionId, folderId);
      // The backend also moves direct Little Tutor children with their parent.
      // Reload the full snapshot so both levels change folders immediately in
      // the UI instead of leaving child rows stale until the next refresh.
      await load(true);
    } catch (error) {
      setFolderError(describeError(error, "Could not move chat"));
    }
  };

  const handleOrganize = useCallback(
    async (sessionId: string, patch: SessionOrganizationPatch) => {
      await updateSessionOrganization(sessionId, patch);
      await load(true);
    },
    [load],
  );

  const HeaderIcon = icon ?? History;
  const headerTitle = title ?? t("Chat History");
  const headerDescription =
    description ??
    t(
      "Browse, organize, rename, delete, and reopen previous conversations from your learning space.",
    );

  return (
    <div className="space-y-6">
      <SpaceSectionHeader
        icon={HeaderIcon}
        title={headerTitle}
        description={headerDescription}
        meta={
          <span className="rounded-full border border-[var(--border)] bg-[var(--card)] px-2 py-0.5 text-[10.5px] font-medium text-[var(--muted-foreground)]">
            {sessions.length} {t("conversations")}
          </span>
        }
        action={
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setCreatingFolder(true);
                setFolderError(null);
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border)]/50 px-3 py-1.5 text-[12px] font-medium text-[var(--muted-foreground)] transition-colors hover:border-[var(--border)] hover:text-[var(--foreground)]"
            >
              <FolderPlus className="h-3.5 w-3.5" />
              {t("New chat folder")}
            </button>
            <button
              type="button"
              onClick={() => void load(true)}
              disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border)]/50 px-3 py-1.5 text-[12px] font-medium text-[var(--muted-foreground)] transition-colors hover:border-[var(--border)] hover:text-[var(--foreground)] disabled:opacity-40"
            >
              {loading ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RefreshCw className="h-3 w-3" />
              )}
              {t("Refresh")}
            </button>
          </div>
        }
      />

      <section className="rounded-2xl border border-[var(--border)] bg-[var(--card)] shadow-sm">
        <div className="space-y-3 border-b border-[var(--border)]/60 px-4 py-3">
          <label className="flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-[13px] text-[var(--muted-foreground)] focus-within:border-[var(--ring)]">
            <Search size={14} strokeWidth={1.7} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t("Search chat history...")}
              className="min-w-0 flex-1 bg-transparent text-[13px] text-[var(--foreground)] outline-none placeholder:text-[var(--muted-foreground)]/55"
            />
          </label>
          {creatingFolder && (
            <div className="flex items-center gap-2">
              <FolderPlus size={15} className="text-[var(--muted-foreground)]" />
              <input
                autoFocus
                value={newFolderName}
                maxLength={50}
                onChange={(event) => setNewFolderName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void handleCreateFolder();
                  if (event.key === "Escape") setCreatingFolder(false);
                }}
                placeholder={t("Chat folder name")}
                className="min-w-0 flex-1 rounded-lg border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-[13px] outline-none focus:border-[var(--ring)]"
              />
              <button
                type="button"
                onClick={() => void handleCreateFolder()}
                disabled={!newFolderName.trim()}
                className="rounded-md p-1.5 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)] disabled:opacity-30"
                aria-label={t("Create folder")}
              >
                <Check size={14} />
              </button>
              <button
                type="button"
                onClick={() => {
                  setCreatingFolder(false);
                  setNewFolderName("");
                }}
                className="rounded-md p-1.5 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)]"
                aria-label={t("Cancel")}
              >
                <X size={14} />
              </button>
            </div>
          )}

          {folderError && (
            <p className="text-[12px] text-[var(--destructive)]">{folderError}</p>
          )}
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            <label className="sr-only" htmlFor="history-course-filter">
              {t("Filter by course")}
            </label>
            <select
              id="history-course-filter"
              value={courseFilter}
              onChange={(event) => setCourseFilter(event.target.value)}
              className="rounded-lg border border-[var(--border)] bg-[var(--background)] px-2.5 py-1.5 text-[12px] text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
            >
              <option value="all">{t("All courses")}</option>
              <option value="unclassified">{t("Unclassified")}</option>
              {courses.map((course) => (
                <option key={course.id} value={course.id}>
                  {course.name}
                </option>
              ))}
            </select>
            <label className="sr-only" htmlFor="history-kind-filter">
              {t("Filter by conversation type")}
            </label>
            <select
              id="history-kind-filter"
              value={kindFilter}
              onChange={(event) => setKindFilter(event.target.value)}
              className="rounded-lg border border-[var(--border)] bg-[var(--background)] px-2.5 py-1.5 text-[12px] text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
            >
              <option value="all">{t("All conversation types")}</option>
              <option value="chat">{t("Main conversations")}</option>
              <option value="selection_tutor">{t("Tutor threads")}</option>
            </select>
            <label className="sr-only" htmlFor="history-archive-filter">
              {t("Filter by archive status")}
            </label>
            <select
              id="history-archive-filter"
              value={archiveFilter}
              onChange={(event) => setArchiveFilter(event.target.value)}
              className="rounded-lg border border-[var(--border)] bg-[var(--background)] px-2.5 py-1.5 text-[12px] text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
            >
              <option value="active">{t("Active")}</option>
              <option value="archived">{t("Archived")}</option>
              <option value="all">{t("Active and archived")}</option>
            </select>
          </div>
        </div>

        <div className="space-y-5 px-3 py-4">
          {loading ? (
            <div className="space-y-2 p-2">
              {[0, 1, 2, 3].map((item) => (
                <div
                  key={item}
                  className="h-8 animate-pulse rounded bg-[var(--muted)]/45"
                />
              ))}
            </div>
          ) : (
            groups.map((group) => {
              const groupKey = group.id ?? "uncategorized";
              const collapsed = collapsedFolderIds.has(groupKey);
              const folder = group.id
                ? folders.find((item) => item.id === group.id)
                : null;
              return (
                <section key={groupKey}>
                  <div className="mb-2 flex min-h-7 items-center gap-2 px-2">
                    <button
                      type="button"
                      onClick={() =>
                        setCollapsedFolderIds((current) => {
                          const next = new Set(current);
                          if (next.has(groupKey)) next.delete(groupKey);
                          else next.add(groupKey);
                          return next;
                        })
                      }
                      className="rounded p-0.5 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)]"
                      aria-expanded={!collapsed}
                      aria-label={t(
                        collapsed ? "Expand folder" : "Collapse folder",
                      )}
                    >
                      <ChevronDown
                        size={13}
                        className={`transition-transform ${collapsed ? "-rotate-90" : ""}`}
                      />
                    </button>
                    <Folder
                      size={15}
                      className="shrink-0 text-[var(--muted-foreground)]"
                    />
                    {editingFolderId === group.id && group.id ? (
                      <input
                        autoFocus
                        value={folderDraft}
                        maxLength={50}
                        onChange={(event) => setFolderDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter")
                            void handleRenameFolder(group.id as string);
                          if (event.key === "Escape") setEditingFolderId(null);
                        }}
                        className="min-w-0 flex-1 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-[12px] outline-none focus:border-[var(--ring)]"
                      />
                    ) : (
                      <h3 className="min-w-0 flex-1 truncate text-[12px] font-semibold text-[var(--foreground)]">
                        {group.name}
                      </h3>
                    )}
                    <span className="text-[11px] text-[var(--muted-foreground)]">
                      {group.sessions.length}
                    </span>
                    {folder && editingFolderId === folder.id && (
                      <button
                        type="button"
                        onClick={() => void handleRenameFolder(folder.id)}
                        className="rounded p-1 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)]"
                        aria-label={t("Save folder name")}
                      >
                        <Check size={12} />
                      </button>
                    )}
                    {folder && editingFolderId !== folder.id && (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingFolderId(folder.id);
                            setFolderDraft(folder.name);
                            setFolderError(null);
                          }}
                          className="rounded p-1 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)]"
                          aria-label={t("Rename folder")}
                        >
                          <Pencil size={12} />
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDeleteFolder(folder)}
                          className="rounded p-1 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--destructive)]"
                          aria-label={t("Delete folder")}
                        >
                          <Trash2 size={12} />
                        </button>
                      </>
                    )}
                  </div>
                  {!collapsed && group.sessions.length > 0 ? (
                    <OrganizedSessionList
                      sessions={group.sessions}
                      courses={courses}
                      activeSessionId={activeSessionId}
                      onSelect={handleSelect}
                      onRename={handleRename}
                      onDelete={handleDelete}
                      onOrganize={handleOrganize}
                      folderOptions={folders}
                      onMove={handleMove}
                    />
                  ) : !collapsed ? (
                    <div className="rounded-lg border border-dashed border-[var(--border)]/60 px-3 py-4 text-center text-[11px] text-[var(--muted-foreground)]/70">
                      {query.trim()
                        ? t("No matching conversations")
                        : t("No conversations in this folder")}
                    </div>
                  ) : null}
                </section>
              );
            })
          )}
        </div>
      </section>
    </div>
  );
}
