import type { SessionFolder, SessionSummary } from "@/lib/session-api";

export interface SessionFolderGroup {
  folder: SessionFolder | null;
  sessions: SessionSummary[];
}

function byPriority(a: SessionSummary, b: SessionSummary): number {
  const pinned =
    Number(Boolean(b.preferences?.pinned)) -
    Number(Boolean(a.preferences?.pinned));
  return pinned || b.updated_at - a.updated_at;
}

/** Build a render-safe tree even if legacy organization data contains a cycle. */
export function organizeSessionTree(
  sessions: SessionSummary[],
  nested: boolean,
): {
  roots: SessionSummary[];
  childrenByParent: Map<string, SessionSummary[]>;
} {
  const byId = new Map(
    sessions.map((session) => [session.session_id, session]),
  );
  const childrenByParent = new Map<string, SessionSummary[]>();
  const roots: SessionSummary[] = [];

  for (const session of sessions) {
    const proposedParent = String(session.preferences?.parent_session_id || "");
    let parentId = nested && byId.has(proposedParent) ? proposedParent : "";
    if (parentId) {
      const visited = new Set([session.session_id]);
      let cursor = parentId;
      while (cursor && byId.has(cursor)) {
        if (visited.has(cursor)) {
          parentId = "";
          break;
        }
        visited.add(cursor);
        cursor = String(byId.get(cursor)?.preferences?.parent_session_id || "");
      }
    }

    if (!parentId) {
      roots.push(session);
      continue;
    }
    const children = childrenByParent.get(parentId) ?? [];
    children.push(session);
    childrenByParent.set(parentId, children);
  }

  roots.sort(byPriority);
  for (const children of childrenByParent.values()) children.sort(byPriority);
  return { roots, childrenByParent };
}

/** Group a bounded session list into known one-level folders plus Uncategorized. */
export function groupSessionsByFolder(
  sessions: SessionSummary[],
  folders: SessionFolder[],
): SessionFolderGroup[] {
  const knownFolderIds = new Set(folders.map((folder) => folder.id));
  const sessionsByFolder = new Map<string, SessionSummary[]>();
  const uncategorized: SessionSummary[] = [];

  for (const session of sessions) {
    const folderId = session.folder_id || "";
    if (!folderId || !knownFolderIds.has(folderId)) {
      uncategorized.push(session);
      continue;
    }
    const grouped = sessionsByFolder.get(folderId) ?? [];
    grouped.push(session);
    sessionsByFolder.set(folderId, grouped);
  }

  return [
    ...folders.map((folder) => ({
      folder,
      sessions: sessionsByFolder.get(folder.id) ?? [],
    })),
    { folder: null, sessions: uncategorized },
  ];
}

/** Build the sidebar's two sibling sections without duplicating conversations. */
export function buildSidebarSessionSections(
  sessions: SessionSummary[],
  folders: SessionFolder[],
  recentLimit = 8,
): {
  folderGroups: SessionFolderGroup[];
  recentSessions: SessionSummary[];
} {
  const groups = groupSessionsByFolder(sessions, folders);
  return {
    folderGroups: groups.filter((group) => group.folder !== null),
    recentSessions: (
      groups.find((group) => group.folder === null)?.sessions ?? []
    ).slice(0, Math.max(0, recentLimit)),
  };
}
