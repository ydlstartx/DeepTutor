"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { SidebarShell } from "@/components/sidebar/SidebarShell";
import { LogoutButton } from "@/components/auth/LogoutButton";
import { AdminLink } from "@/components/auth/AdminLink";
import { ProfileLink } from "@/components/auth/ProfileLink";
import { useUnifiedChat } from "@/context/UnifiedChatContext";
import {
  createSessionFolder,
  deleteSessionFolder,
  deleteSession,
  listSessionFolders,
  listSessions,
  moveSessionToFolder,
  renameSessionFolder,
  updateSessionOrganization,
  updateSessionTitle,
  type SessionFolder,
  type SessionOrganizationPatch,
  type SessionSummary,
} from "@/lib/session-api";
import { listCourses, type StudyCourse } from "@/lib/courses-api";

function WorkspaceSidebarImpl() {
  const { t } = useTranslation();
  const router = useRouter();
  const {
    newSession,
    cancelStreamingTurn,
    selectedSessionId,
    sessionStatuses,
    sidebarRefreshToken,
  } = useUnifiedChat();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [courses, setCourses] = useState<StudyCourse[]>([]);
  const [folders, setFolders] = useState<SessionFolder[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const hasLoadedSessionsRef = useRef(false);

  const refreshSessions = useCallback(async () => {
    if (!hasLoadedSessionsRef.current) {
      setLoadingSessions(true);
    }
    try {
      const [nextSessions, nextCourses, nextFolders] = await Promise.all([
        listSessions(50, 0, { force: true }),
        listCourses({ force: true }),
        listSessionFolders({ force: true }),
      ]);
      setSessions(nextSessions);
      setCourses(nextCourses);
      setFolders(nextFolders);
      hasLoadedSessionsRef.current = true;
    } catch (error) {
      console.error("Failed to load sessions", error);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  // First mount shows the skeleton; subsequent refreshes triggered by
  // ``sidebarRefreshToken`` (STREAM_END, server-side session bind,
  // turn deletion) silently swap in the new list. Resetting the ref
  // each refresh briefly re-renders the loading skeleton, which the
  // user perceives as a flicker on every message send / Answer Now.
  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, sidebarRefreshToken]);

  const orderedSessions = useMemo(() => {
    const ordered = sessions
      .map((session, index) => {
        const runtime = sessionStatuses[session.session_id];
        return {
          index,
          session: runtime
            ? {
                ...session,
                status: runtime.status,
                active_turn_id: runtime.activeTurnId || session.active_turn_id,
              }
            : session,
        };
      })
      .sort((a, b) => {
        const aPriority = a.session.status === "running" ? 0 : 1;
        const bPriority = b.session.status === "running" ? 0 : 1;
        if (aPriority !== bPriority) return aPriority - bPriority;
        return a.index - b.index;
      })
      .map(({ session }) => session);
    return ordered;
  }, [sessions, sessionStatuses]);

  // Cancel any in-flight streaming turn before starting a fresh session, so a
  // new chat never inherits a still-running turn (mirrors handleDeleteSession).
  const handleNewChat = useCallback(() => {
    cancelStreamingTurn();
    newSession();
    router.push("/home");
  }, [cancelStreamingTurn, newSession, router]);

  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      router.push(`/home/${sessionId}`);
    },
    [router],
  );

  const handleRenameSession = useCallback(
    async (sessionId: string, title: string) => {
      const updated = await updateSessionTitle(sessionId, title);
      setSessions((prev) =>
        prev.map((session) =>
          session.session_id === sessionId
            ? {
                ...session,
                title: updated.title,
                updated_at: updated.updated_at,
              }
            : session,
        ),
      );
    },
    [],
  );

  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      if (!window.confirm(t("Delete this chat history?"))) return;
      await deleteSession(sessionId);
      setSessions((prev) =>
        prev.filter((session) => session.session_id !== sessionId),
      );
      if (selectedSessionId === sessionId) {
        cancelStreamingTurn();
        newSession();
        router.push("/home");
      }
    },
    [cancelStreamingTurn, newSession, router, selectedSessionId, t],
  );

  const handleOrganizeSession = useCallback(
    async (sessionId: string, patch: SessionOrganizationPatch) => {
      const updated = await updateSessionOrganization(sessionId, patch);
      setSessions((previous) =>
        previous.map((session) =>
          session.session_id === sessionId
            ? {
                ...session,
                updated_at: updated.updated_at,
                preferences: updated.preferences,
              }
            : session,
        ),
      );
    },
    [],
  );

  const handleCreateFolder = useCallback(async (name: string) => {
    const folder = await createSessionFolder(name);
    setFolders((previous) => [...previous, folder]);
  }, []);

  const handleRenameFolder = useCallback(
    async (folderId: string, name: string) => {
      const updated = await renameSessionFolder(folderId, name);
      setFolders((previous) =>
        previous.map((folder) => (folder.id === folderId ? updated : folder)),
      );
    },
    [],
  );

  const handleDeleteFolder = useCallback(
    async (folderId: string) => {
      await deleteSessionFolder(folderId);
      await refreshSessions();
    },
    [refreshSessions],
  );

  const handleMoveSessionToFolder = useCallback(
    async (sessionId: string, folderId: string | null) => {
      await moveSessionToFolder(sessionId, folderId);
      // The backend also moves direct Little Tutor children. Refresh the
      // bounded sidebar snapshot and folder counts as one consistent view.
      await refreshSessions();
    },
    [refreshSessions],
  );

  return (
    <SidebarShell
      showSessions
      sessions={orderedSessions}
      courses={courses}
      folders={folders}
      activeSessionId={selectedSessionId}
      loadingSessions={loadingSessions}
      onNewChat={handleNewChat}
      onSelectSession={handleSelectSession}
      onRenameSession={handleRenameSession}
      onDeleteSession={handleDeleteSession}
      onOrganizeSession={handleOrganizeSession}
      onCreateFolder={handleCreateFolder}
      onRenameFolder={handleRenameFolder}
      onDeleteFolder={handleDeleteFolder}
      onMoveSessionToFolder={handleMoveSessionToFolder}
      footerSlot={(collapsed) => (
        <>
          <ProfileLink collapsed={collapsed} />
          <AdminLink collapsed={collapsed} />
          <LogoutButton collapsed={collapsed} />
        </>
      )}
    />
  );
}

// Memoized: the component has no props and only consumes low-frequency
// context (sessions/statuses — stream events no longer touch it), so memo
// blocks re-renders from unrelated parent updates.
const WorkspaceSidebar = memo(WorkspaceSidebarImpl);
WorkspaceSidebar.displayName = "WorkspaceSidebar";
export default WorkspaceSidebar;
