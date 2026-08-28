"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { SidebarShell } from "@/components/sidebar/SidebarShell";
import { LogoutButton } from "@/components/auth/LogoutButton";
import { AdminLink } from "@/components/auth/AdminLink";
import { ProfileLink } from "@/components/auth/ProfileLink";
import { useAppShell } from "@/context/AppShellContext";
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

export default function UtilitySidebar() {
  const { t } = useTranslation();
  const router = useRouter();
  const { activeSessionId, setActiveSessionId } = useAppShell();
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

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      router.push(`/home/${sessionId}`);
    },
    [router, setActiveSessionId],
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
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
      }
    },
    [activeSessionId, setActiveSessionId, t],
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
      await refreshSessions();
    },
    [refreshSessions],
  );

  return (
    <SidebarShell
      showSessions
      sessions={sessions}
      courses={courses}
      folders={folders}
      activeSessionId={activeSessionId}
      loadingSessions={loadingSessions}
      onNewChat={() => setActiveSessionId(null)}
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
