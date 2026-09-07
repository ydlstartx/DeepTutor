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
import {
  fetchReadingCollectionIndex,
  type ReadingCollectionLabel,
} from "@/lib/reading-workspace-api";
import {
  fetchMasteryTopicIndex,
  type MasteryTopicLabel,
} from "@/lib/learning-api";
import { sessionRoute } from "@/lib/mastery-session";
import { subscribeSessionChanges } from "@/lib/session-events";

export default function UtilitySidebar() {
  const { t } = useTranslation();
  const router = useRouter();
  const { activeSessionId, setActiveSessionId } = useAppShell();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [courses, setCourses] = useState<StudyCourse[]>([]);
  const [folders, setFolders] = useState<SessionFolder[]>([]);
  const [masteryTopics, setMasteryTopics] = useState<MasteryTopicLabel[]>([]);
  const [readingCollections, setReadingCollections] = useState<
    ReadingCollectionLabel[]
  >([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const hasLoadedSessionsRef = useRef(false);

  const refreshSessions = useCallback(async () => {
    if (!hasLoadedSessionsRef.current) {
      setLoadingSessions(true);
    }
    try {
      // Labels only name a heading, so losing them costs grouping, not the list.
      const [nextSessions, nextCourses, nextTopics, nextCollections, nextFolders] =
        await Promise.all([
          listSessions(50, 0, { force: true }),
          listCourses({ force: true }),
          fetchMasteryTopicIndex().catch(() => [] as MasteryTopicLabel[]),
          fetchReadingCollectionIndex(),
          listSessionFolders({ force: true }),
        ]);
      setSessions(nextSessions);
      setCourses(nextCourses);
      setMasteryTopics(nextTopics);
      setReadingCollections(nextCollections);
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

  // A conversation can be archived, restored or deleted from a route the
  // sidebar knows nothing about — Settings › Archive being the reason this
  // exists. Without it the list keeps hiding a conversation that was just
  // restored two panes away.
  useEffect(
    () => subscribeSessionChanges(() => void refreshSessions()),
    [refreshSessions],
  );

  // A study conversation opens on its own path — see ``sessionRoute``.
  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      const session = sessions.find((item) => item.session_id === sessionId);
      router.push(session ? sessionRoute(session) : `/chat/${sessionId}`);
    },
    [router, sessions, setActiveSessionId],
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
      masteryTopics={masteryTopics}
      readingCollections={readingCollections}
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
