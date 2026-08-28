"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createPortal } from "react-dom";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useAppShell } from "@/context/AppShellContext";
import {
  BookOpen,
  BookText,
  Bot,
  Brain,
  Check,
  ChevronDown,
  Folder,
  FolderPlus,
  Github,
  HeartHandshake,
  House,
  LayoutGrid,
  Library,
  Lock,
  MoreHorizontal,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  PenLine,
  Settings,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import OrganizedSessionList from "@/components/courses/OrganizedSessionList";
import SessionList from "@/components/SessionList";
import { useSidebarDrawer } from "@/components/layout/AppShell";
import { useDevice } from "@/hooks/useDevice";
import { VersionBadge } from "@/components/sidebar/VersionBadge";
import type {
  SessionFolder,
  SessionOrganizationPatch,
  SessionSummary,
} from "@/lib/session-api";
import { buildSidebarSessionSections } from "@/lib/session-organization";
import type { StudyCourse } from "@/lib/courses-api";
import { Tooltip } from "@/components/ui/Tooltip";
import { useCapabilityAccess } from "@/components/access/CapabilityAccessContext";
import type { Capability } from "@/lib/capability-routes";
import { PUBLIC_PRODUCT_NAME } from "@/lib/public-brand";
import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  normalizeSidebarWidth,
} from "@/context/app-shell-storage";

interface NavEntry {
  href: string;
  label: string;
  icon: LucideIcon;
  tooltipKey?: string;
  /** Model capability this feature needs; locked when the user lacks it. */
  requires?: Capability;
}

const PRIMARY_NAV: NavEntry[] = [
  {
    href: "/home",
    label: "Home",
    icon: House,
    tooltipKey: "Home tooltip",
    requires: "llm",
  },
  {
    href: "/partners",
    label: "Partners",
    icon: HeartHandshake,
    tooltipKey: "Partners tooltip",
    requires: "llm",
  },
  {
    // My Agents is its own top-level feature (pulled out of the Learning
    // Space): connect a live local Claude Code / Codex to consult in chat,
    // and manage imported agent conversations. Ungated — managing connections
    // and imports needs no per-user model grant.
    href: "/agents",
    label: "My Agents",
    icon: Bot,
    tooltipKey: "Agents tooltip",
  },
  {
    href: "/co-writer",
    label: "Co-Writer",
    icon: PenLine,
    tooltipKey: "Co-Writer tooltip",
    requires: "llm",
  },
  {
    href: "/book",
    label: "Book",
    icon: Library,
    tooltipKey: "Book tooltip",
    requires: "llm",
  },
  {
    href: "/space",
    label: "Learning Space",
    icon: LayoutGrid,
    tooltipKey: "Space tooltip",
  },
];

const SECONDARY_NAV: NavEntry[] = [
  {
    // Memory is its own top-level console (pulled out of the Learning Space):
    // a place to inspect and curate the tutor's long-term memory, not a daily
    // workspace. Never gated — memory has no per-user model requirement.
    href: "/memory",
    label: "Memory",
    icon: Brain,
    tooltipKey: "Memory tooltip",
  },
  {
    // Knowledge Center sits just above Settings: it's a console for managing
    // KBs and retrieval engines, not a daily workspace. Never gated — embedding
    // / search are shared admin infrastructure, no per-user model grant needed.
    href: "/knowledge",
    label: "Knowledge Center",
    icon: BookOpen,
    tooltipKey: "Knowledge tooltip",
  },
  { href: "/settings", label: "Settings", icon: Settings },
];
const GITHUB_REPO_URL = "https://github.com/HKUDS/DeepTutor";
const DOCS_URL = "https://deeptutor.info/";
const FOLDER_MENU_WIDTH = 176;

interface FolderMenuPosition {
  left: number;
  top: number;
  openUpward: boolean;
}

function placeFolderMenu(anchor: DOMRect): FolderMenuPosition {
  const margin = 12;
  const gap = 4;
  const estimatedHeight = 76;
  const openUpward = window.innerHeight - anchor.bottom < estimatedHeight + gap;
  return {
    left: Math.max(
      margin,
      Math.min(
        anchor.right - FOLDER_MENU_WIDTH,
        window.innerWidth - FOLDER_MENU_WIDTH - margin,
      ),
    ),
    top: openUpward ? anchor.top - gap : anchor.bottom + gap,
    openUpward,
  };
}

interface SidebarShellProps {
  sessions?: SessionSummary[];
  activeSessionId?: string | null;
  loadingSessions?: boolean;
  showSessions?: boolean;
  /** Clicking the Chat nav item resets to a fresh session via this handler. */
  onNewChat?: () => void;
  onSelectSession?: (sessionId: string) => void | Promise<void>;
  onRenameSession?: (sessionId: string, title: string) => void | Promise<void>;
  onDeleteSession?: (sessionId: string) => void | Promise<void>;
  courses?: StudyCourse[];
  folders?: SessionFolder[];
  onOrganizeSession?: (
    sessionId: string,
    patch: SessionOrganizationPatch,
  ) => void | Promise<void>;
  onCreateFolder?: (name: string) => void | Promise<void>;
  onRenameFolder?: (
    folderId: string,
    name: string,
  ) => void | Promise<void>;
  onDeleteFolder?: (folderId: string) => void | Promise<void>;
  onMoveSessionToFolder?: (
    sessionId: string,
    folderId: string | null,
  ) => void | Promise<void>;
  /**
   * Footer content rendered below the nav. Pass a render function to receive
   * the current ``collapsed`` state so footer items (e.g. Admin / Sign out) can
   * switch to their icon-only variant when the rail is collapsed.
   */
  footerSlot?: ReactNode | ((collapsed: boolean) => ReactNode);
}

export function SidebarShell({
  sessions = [],
  activeSessionId = null,
  loadingSessions = false,
  showSessions = false,
  onNewChat,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  courses = [],
  folders = [],
  onOrganizeSession,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  onMoveSessionToFolder,
  footerSlot,
}: SidebarShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { t } = useTranslation();
  const { has } = useCapabilityAccess();
  const {
    sidebarCollapsed,
    setSidebarCollapsed: setCollapsed,
    sidebarWidth,
    setSidebarWidth,
  } = useAppShell();
  const { isMobile } = useDevice();
  const drawer = useSidebarDrawer();

  // Inside the mobile drawer the icon-only rail is pointless — the panel is
  // already hidden when you don't want it, so it always opens fully expanded
  // regardless of the persisted desktop preference.
  const collapsed = sidebarCollapsed && !isMobile;

  /** Dismiss the drawer on nav clicks that actually navigate in-place. */
  const closeDrawerOnNav = (event: React.MouseEvent) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1)
      return;
    drawer?.close();
  };

  const navLocked = (item: NavEntry) =>
    item.requires ? !has(item.requires) : false;
  const lockedTooltip = t("Locked — contact your administrator to get access.");
  const renderedFooter =
    typeof footerSlot === "function" ? footerSlot(collapsed) : footerSlot;
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [folderError, setFolderError] = useState<string | null>(null);
  const [openFolderMenuId, setOpenFolderMenuId] = useState<string | null>(null);
  const [folderMenuPosition, setFolderMenuPosition] =
    useState<FolderMenuPosition | null>(null);
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null);
  const [folderDraft, setFolderDraft] = useState("");
  const [collapsedFolderIds, setCollapsedFolderIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [draftSidebarWidth, setDraftSidebarWidth] = useState(sidebarWidth);
  const [resizingSidebar, setResizingSidebar] = useState(false);
  const resizeAbortRef = useRef<AbortController | null>(null);
  const folderMenuRootRef = useRef<HTMLDivElement>(null);
  const folderMenuAnchorRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!resizingSidebar) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDraftSidebarWidth(sidebarWidth);
    }
  }, [resizingSidebar, sidebarWidth]);

  useEffect(
    () => () => {
      resizeAbortRef.current?.abort();
      if (typeof document !== "undefined") {
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    },
    [],
  );

  useEffect(() => {
    if (!openFolderMenuId) return;
    const closeMenu = () => {
      setOpenFolderMenuId(null);
      setFolderMenuPosition(null);
    };
    const closeOnOutsideClick = (event: MouseEvent) => {
      const target = event.target as Node;
      if (
        !folderMenuRootRef.current?.contains(target) &&
        !folderMenuAnchorRef.current?.contains(target)
      ) {
        closeMenu();
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeMenu();
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", closeMenu);
    window.addEventListener("scroll", closeMenu, true);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("resize", closeMenu);
      window.removeEventListener("scroll", closeMenu, true);
    };
  }, [openFolderMenuId]);

  const handleHomeClick = (event: React.MouseEvent) => {
    // Always reset to a fresh session (mirrors the old "New Chat" affordance);
    // let modifier-clicks fall through to default Link behavior so middle-click
    // open-in-new-tab still works.
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1)
      return;
    event.preventDefault();
    drawer?.close();
    onNewChat?.();
    router.push("/home");
  };

  const startSidebarResize = (event: React.PointerEvent<HTMLDivElement>) => {
    if (isMobile || event.button !== 0) return;
    event.preventDefault();
    resizeAbortRef.current?.abort();
    const controller = new AbortController();
    resizeAbortRef.current = controller;
    const startX = event.clientX;
    const startWidth = draftSidebarWidth;
    let latestWidth = startWidth;
    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    setResizingSidebar(true);

    const move = (moveEvent: PointerEvent) => {
      latestWidth = normalizeSidebarWidth(
        startWidth + moveEvent.clientX - startX,
      );
      setDraftSidebarWidth(latestWidth);
    };
    const finish = () => {
      controller.abort();
      if (resizeAbortRef.current === controller) resizeAbortRef.current = null;
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
      setResizingSidebar(false);
      setSidebarWidth(latestWidth);
    };

    window.addEventListener("pointermove", move, { signal: controller.signal });
    window.addEventListener("pointerup", finish, {
      once: true,
      signal: controller.signal,
    });
    window.addEventListener("pointercancel", finish, {
      once: true,
      signal: controller.signal,
    });
  };

  const handleResizeKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    let nextWidth: number | null = null;
    if (event.key === "ArrowLeft") nextWidth = draftSidebarWidth - 10;
    if (event.key === "ArrowRight") nextWidth = draftSidebarWidth + 10;
    if (event.key === "Home") nextWidth = MIN_SIDEBAR_WIDTH;
    if (event.key === "End") nextWidth = MAX_SIDEBAR_WIDTH;
    if (nextWidth === null) return;
    event.preventDefault();
    const normalized = normalizeSidebarWidth(nextWidth);
    setDraftSidebarWidth(normalized);
    setSidebarWidth(normalized);
  };

  const sidebarSessions = sessions.filter(
    (session) =>
      !session.preferences?.archived &&
      !session.preferences?.parent_session_id,
  );
  const folderManagementEnabled = Boolean(
    onCreateFolder && onMoveSessionToFolder,
  );
  const folderActionsEnabled = Boolean(onRenameFolder && onDeleteFolder);
  const { folderGroups: sidebarFolderGroups, recentSessions } =
    buildSidebarSessionSections(sidebarSessions, folders);

  const handleCreateFolder = async () => {
    const name = newFolderName.trim();
    if (!name || !onCreateFolder) return;
    setFolderError(null);
    try {
      await onCreateFolder(name);
      setNewFolderName("");
      setCreatingFolder(false);
    } catch (error) {
      setFolderError(
        error instanceof Error
          ? t(error.message)
          : t("Could not create folder"),
      );
    }
  };

  const handleMoveSessionToFolder = async (
    sessionId: string,
    folderId: string | null,
  ) => {
    if (!onMoveSessionToFolder) return;
    setFolderError(null);
    try {
      await onMoveSessionToFolder(sessionId, folderId);
    } catch (error) {
      setFolderError(
        error instanceof Error ? t(error.message) : t("Could not move chat"),
      );
    }
  };

  const handleRenameFolder = async (folder: SessionFolder) => {
    const name = folderDraft.trim();
    if (!name || !onRenameFolder) return;
    if (name === folder.name) {
      setEditingFolderId(null);
      setFolderDraft("");
      return;
    }
    setFolderError(null);
    try {
      await onRenameFolder(folder.id, name);
      setEditingFolderId(null);
      setFolderDraft("");
    } catch (error) {
      setFolderError(
        error instanceof Error
          ? t(error.message)
          : t("Could not rename folder"),
      );
    }
  };

  const handleDeleteFolder = async (folder: SessionFolder) => {
    if (!onDeleteFolder) return;
    if (
      !window.confirm(
        t(
          'Delete folder "{{name}}"? Its conversations will move to Uncategorized.',
          { name: folder.name },
        ),
      )
    )
      return;
    setOpenFolderMenuId(null);
    setFolderMenuPosition(null);
    setFolderError(null);
    try {
      await onDeleteFolder(folder.id);
      if (editingFolderId === folder.id) {
        setEditingFolderId(null);
        setFolderDraft("");
      }
    } catch (error) {
      setFolderError(
        error instanceof Error
          ? t(error.message)
          : t("Could not delete folder"),
      );
    }
  };

  /* ---- Collapsed state ---- */
  if (collapsed) {
    return (
      <aside className="group/sb relative flex h-dvh w-[60px] shrink-0 flex-col items-center bg-[var(--secondary)] py-3 transition-all duration-200">
        {/* Header: logo + collapse toggle (toggle replaces logo on hover) */}
        <div className="relative mb-2 flex h-9 w-9 items-center justify-center">
          <Link
            href="/"
            aria-label={PUBLIC_PRODUCT_NAME}
            className="flex items-center justify-center transition-opacity duration-150 group-hover/sb:opacity-0"
          >
            <BookOpen className="h-[22px] w-[22px]" strokeWidth={1.7} />
          </Link>
          <button
            onClick={() => setCollapsed(false)}
            className="absolute inset-0 flex items-center justify-center rounded-lg text-[var(--muted-foreground)] opacity-0 transition-all duration-150 hover:bg-[var(--background)]/60 hover:text-[var(--foreground)] group-hover/sb:opacity-100"
            aria-label={t("Expand sidebar")}
          >
            <PanelLeftOpen size={16} />
          </button>
        </div>

        {/* Primary nav */}
        <nav className="mt-1 flex w-full flex-col items-center gap-1 px-1.5">
          {PRIMARY_NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            const locked = navLocked(item);
            const description = locked
              ? lockedTooltip
              : item.tooltipKey
                ? t(item.tooltipKey)
                : undefined;
            if (locked) {
              return (
                <Tooltip
                  key={item.href}
                  label={t(item.label)}
                  description={description}
                  side="right"
                >
                  <div
                    aria-label={`${t(item.label)} — ${lockedTooltip}`}
                    aria-disabled
                    className="relative flex h-9 w-9 cursor-not-allowed items-center justify-center rounded-xl text-[var(--muted-foreground)]/40"
                  >
                    <item.icon size={18} strokeWidth={1.6} />
                    <Lock
                      size={10}
                      strokeWidth={2}
                      className="absolute bottom-1 right-1 text-[var(--muted-foreground)]/70"
                    />
                  </div>
                </Tooltip>
              );
            }
            return (
              <Tooltip
                key={item.href}
                label={t(item.label)}
                description={description}
                side="right"
              >
                <Link
                  href={item.href}
                  onClick={item.href === "/home" ? handleHomeClick : undefined}
                  aria-label={t(item.label)}
                  className={`relative flex h-9 w-9 items-center justify-center rounded-xl transition-all duration-150 ${
                    active
                      ? "bg-[var(--accent)] text-[var(--foreground)] shadow-sm"
                      : "text-[var(--foreground)]/85 hover:bg-[var(--background)]/60 hover:text-[var(--foreground)]"
                  }`}
                >
                  <item.icon size={18} strokeWidth={active ? 2 : 1.6} />
                </Link>
              </Tooltip>
            );
          })}
        </nav>

        <div className="flex-1" />

        {/* Secondary nav + footer */}
        <div className="flex w-full flex-col items-center gap-1 px-1.5">
          <div className="my-1 h-px w-7 bg-[var(--border)]/40" />
          {SECONDARY_NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                title={t(item.label) as string}
                className={`relative flex h-9 w-9 items-center justify-center rounded-xl transition-all duration-150 ${
                  active
                    ? "bg-[var(--accent)] text-[var(--foreground)] shadow-sm"
                    : "text-[var(--foreground)]/85 hover:bg-[var(--background)]/60 hover:text-[var(--foreground)]"
                }`}
              >
                <item.icon size={18} strokeWidth={active ? 2 : 1.6} />
              </Link>
            );
          })}
          {renderedFooter}
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noreferrer noopener"
            title={t("Docs") as string}
            aria-label={t("Docs") as string}
            className="mt-1 flex h-9 w-9 items-center justify-center rounded-xl text-[var(--muted-foreground)]/70 transition-colors hover:bg-[var(--background)]/50 hover:text-[var(--foreground)]"
          >
            <BookText size={15} strokeWidth={1.6} />
          </a>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            title="GitHub"
            aria-label="GitHub"
            className="flex h-9 w-9 items-center justify-center rounded-xl text-[var(--muted-foreground)]/70 transition-colors hover:bg-[var(--background)]/50 hover:text-[var(--foreground)]"
          >
            <Github size={15} strokeWidth={1.6} />
          </a>
          <VersionBadge collapsed />
        </div>
      </aside>
    );
  }

  /* ---- Expanded state ---- */
  return (
    <aside
      className={`relative flex h-dvh shrink-0 flex-col bg-[var(--secondary)] ${
        resizingSidebar ? "" : "transition-[width] duration-200"
      }`}
      style={{ width: isMobile ? DEFAULT_SIDEBAR_WIDTH : draftSidebarWidth }}
    >
      {!isMobile && (
        <div
          role="separator"
          aria-label={t("Resize sidebar")}
          aria-orientation="vertical"
          aria-valuemin={MIN_SIDEBAR_WIDTH}
          aria-valuemax={MAX_SIDEBAR_WIDTH}
          aria-valuenow={draftSidebarWidth}
          tabIndex={0}
          onPointerDown={startSidebarResize}
          onKeyDown={handleResizeKeyDown}
          onDoubleClick={() => {
            setDraftSidebarWidth(DEFAULT_SIDEBAR_WIDTH);
            setSidebarWidth(DEFAULT_SIDEBAR_WIDTH);
          }}
          className="group/resize absolute -right-1 top-0 z-30 h-full w-2 cursor-col-resize touch-none outline-none after:absolute after:bottom-0 after:left-1/2 after:top-0 after:w-px after:-translate-x-1/2 after:bg-transparent after:transition-colors hover:after:bg-[var(--primary)]/45 focus-visible:after:bg-[var(--primary)]/60"
        />
      )}
      {/* Header: logo + collapse toggle */}
      <div className="flex h-14 items-center justify-between px-4">
        <Link
          href="/"
          aria-label={PUBLIC_PRODUCT_NAME}
          className="group flex items-center gap-1.5"
        >
          <BookOpen
            className="h-[22px] w-[22px] transition-transform duration-200 group-hover:scale-105"
            strokeWidth={1.7}
          />
          <span className="font-serif text-[15px] font-semibold tracking-wide">
            {PUBLIC_PRODUCT_NAME}
          </span>
        </Link>
        {/* The rail is a desktop affordance; in the drawer the scrim and the
            top-bar toggle already own "make this go away". */}
        <button
          onClick={() => setCollapsed(true)}
          className="rounded-md p-1 text-[var(--muted-foreground)] transition-colors hover:text-[var(--foreground)] max-md:hidden"
          aria-label={t("Collapse sidebar")}
        >
          <PanelLeftClose size={15} />
        </button>
      </div>

      {/* Primary nav */}
      <nav className="px-2 pt-1">
        <div className="space-y-px">
          {PRIMARY_NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            const locked = navLocked(item);
            if (locked) {
              return (
                <Tooltip
                  key={item.href}
                  label={t(item.label)}
                  description={lockedTooltip}
                  side="right"
                >
                  <div
                    aria-label={`${t(item.label)} — ${lockedTooltip}`}
                    aria-disabled
                    className="flex cursor-not-allowed items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] text-[var(--muted-foreground)]/40"
                  >
                    <item.icon size={16} strokeWidth={1.5} />
                    <span>{t(item.label)}</span>
                    <Lock size={13} strokeWidth={1.8} className="ml-auto" />
                  </div>
                </Tooltip>
              );
            }
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={
                  item.href === "/home" ? handleHomeClick : closeDrawerOnNav
                }
                className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] transition-colors ${
                  active
                    ? "bg-[var(--accent)] font-medium text-[var(--foreground)]"
                    : "text-[var(--foreground)]/85 hover:bg-[var(--background)]/60 hover:text-[var(--foreground)]"
                }`}
              >
                <item.icon size={16} strokeWidth={active ? 1.9 : 1.5} />
                <span>{t(item.label)}</span>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Chat history — its own region below the nav, takes remaining height */}
      {showSessions && onSelectSession && onRenameSession && onDeleteSession ? (
        <section className="mt-4 flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
            <div className="flex items-center px-2 py-1 text-[11.5px] font-normal text-[var(--muted-foreground)]/60">
              <span className="min-w-0 flex-1">{t("Chat folders")}</span>
              {onCreateFolder ? (
                <button
                  type="button"
                  onClick={() => {
                    setCreatingFolder(true);
                    setFolderError(null);
                  }}
                  className="rounded p-1 opacity-65 transition-opacity hover:bg-[var(--background)] hover:opacity-100 focus:opacity-100"
                  title={t("New chat folder") as string}
                  aria-label={t("New chat folder")}
                >
                  <FolderPlus size={13} strokeWidth={1.7} />
                </button>
              ) : null}
            </div>
            {creatingFolder ? (
              <div className="mb-1.5 flex items-center gap-1 rounded-lg border border-[var(--border)]/60 bg-[var(--background)]/35 p-1">
                <input
                  autoFocus
                  value={newFolderName}
                  maxLength={50}
                  onChange={(event) => setNewFolderName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void handleCreateFolder();
                    if (event.key === "Escape") {
                      setCreatingFolder(false);
                      setNewFolderName("");
                      setFolderError(null);
                    }
                  }}
                  placeholder={t("Chat folder name")}
                  className="min-w-0 flex-1 bg-transparent px-1.5 py-0.5 text-[11.5px] text-[var(--foreground)] outline-none placeholder:text-[var(--muted-foreground)]/55"
                />
                <button
                  type="button"
                  onClick={() => void handleCreateFolder()}
                  disabled={!newFolderName.trim()}
                  className="rounded p-1 hover:bg-[var(--background)] disabled:opacity-30"
                  aria-label={t("Create folder")}
                >
                  <Check size={11} />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCreatingFolder(false);
                    setNewFolderName("");
                    setFolderError(null);
                  }}
                  className="rounded p-1 hover:bg-[var(--background)]"
                  aria-label={t("Cancel")}
                >
                  <X size={11} />
                </button>
              </div>
            ) : null}
            {folderError ? (
              <p className="mb-1.5 px-1 text-[10.5px] text-[var(--destructive)]">
                {folderError}
              </p>
            ) : null}
            {!loadingSessions &&
            onOrganizeSession &&
            folderManagementEnabled ? (
              <div className="space-y-1">
                {sidebarFolderGroups.map((group) => {
                  const folder = group.folder;
                  if (!folder) return null;
                  const collapsedGroup = collapsedFolderIds.has(folder.id);
                  return (
                    <section key={folder.id}>
                      {editingFolderId === folder.id ? (
                        <div className="flex items-center gap-1 rounded-lg bg-[var(--background)]/35 p-1">
                          <Folder
                            size={12}
                            className="ml-1 shrink-0 text-[var(--muted-foreground)]"
                          />
                          <input
                            autoFocus
                            value={folderDraft}
                            maxLength={50}
                            onChange={(event) =>
                              setFolderDraft(event.target.value)
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter")
                                void handleRenameFolder(folder);
                              if (event.key === "Escape") {
                                setEditingFolderId(null);
                                setFolderDraft("");
                                setFolderError(null);
                              }
                            }}
                            aria-label={t("Chat folder name")}
                            className="min-w-0 flex-1 bg-transparent px-1 py-0.5 text-[11.5px] text-[var(--foreground)] outline-none"
                          />
                          <button
                            type="button"
                            onClick={() => void handleRenameFolder(folder)}
                            disabled={!folderDraft.trim()}
                            className="rounded p-1 hover:bg-[var(--background)] disabled:opacity-30"
                            aria-label={t("Save folder name")}
                          >
                            <Check size={11} />
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setEditingFolderId(null);
                              setFolderDraft("");
                              setFolderError(null);
                            }}
                            className="rounded p-1 hover:bg-[var(--background)]"
                            aria-label={t("Cancel")}
                          >
                            <X size={11} />
                          </button>
                        </div>
                      ) : (
                        <div className="group/folder relative flex min-w-0 items-center">
                          <button
                            type="button"
                            onClick={() =>
                              setCollapsedFolderIds((current) => {
                                const next = new Set(current);
                                if (next.has(folder.id)) next.delete(folder.id);
                                else next.add(folder.id);
                                return next;
                              })
                            }
                            className="flex min-w-0 flex-1 items-center gap-1.5 rounded-lg px-2 py-1.5 pr-8 text-left text-[12.5px] text-[var(--muted-foreground)] transition-colors hover:bg-[var(--background)]/40 hover:text-[var(--foreground)]"
                            aria-expanded={!collapsedGroup}
                            aria-label={t(
                              collapsedGroup
                                ? "Expand folder"
                                : "Collapse folder",
                            )}
                          >
                            <ChevronDown
                              size={11}
                              className={`shrink-0 transition-transform ${collapsedGroup ? "-rotate-90" : ""}`}
                            />
                            <Folder size={13} className="shrink-0" />
                            <span className="min-w-0 flex-1 truncate">
                              {folder.name}
                            </span>
                          </button>
                          {folderActionsEnabled ? (
                            <button
                              type="button"
                              onClick={(event) => {
                                if (openFolderMenuId === folder.id) {
                                  setOpenFolderMenuId(null);
                                  setFolderMenuPosition(null);
                                  return;
                                }
                                folderMenuAnchorRef.current =
                                  event.currentTarget;
                                setFolderMenuPosition(
                                  placeFolderMenu(
                                    event.currentTarget.getBoundingClientRect(),
                                  ),
                                );
                                setOpenFolderMenuId(folder.id);
                              }}
                              className={`absolute right-1 rounded p-1 transition-opacity hover:bg-[var(--muted)] ${
                                openFolderMenuId === folder.id
                                  ? "opacity-100"
                                  : "opacity-0 group-hover/folder:opacity-100 focus:opacity-100"
                              }`}
                              aria-label={t("Folder actions")}
                              aria-haspopup="menu"
                              aria-expanded={openFolderMenuId === folder.id}
                            >
                              <MoreHorizontal size={13} />
                            </button>
                          ) : null}
                          {openFolderMenuId === folder.id &&
                          folderMenuPosition &&
                          typeof document !== "undefined"
                            ? createPortal(
                                <div
                                  ref={folderMenuRootRef}
                                  role="menu"
                                  style={{
                                    left: folderMenuPosition.left,
                                    top: folderMenuPosition.top,
                                    transform: folderMenuPosition.openUpward
                                      ? "translateY(-100%)"
                                      : undefined,
                                    transformOrigin:
                                      folderMenuPosition.openUpward
                                        ? "bottom"
                                        : "top",
                                  }}
                                  className="fixed z-[100] w-44 rounded-xl border border-[var(--border)] bg-[var(--popover)] p-1.5 text-[12px] shadow-xl"
                                >
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() => {
                                      setEditingFolderId(folder.id);
                                      setFolderDraft(folder.name);
                                      setOpenFolderMenuId(null);
                                      setFolderMenuPosition(null);
                                      setFolderError(null);
                                    }}
                                    className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[var(--foreground)] hover:bg-[var(--muted)]"
                                  >
                                    <Pencil size={13} />
                                    <span>{t("Rename folder")}</span>
                                  </button>
                                  <button
                                    type="button"
                                    role="menuitem"
                                    onClick={() =>
                                      void handleDeleteFolder(folder)
                                    }
                                    className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[var(--destructive)] hover:bg-[var(--muted)]"
                                  >
                                    <Trash2 size={13} />
                                    <span>{t("Delete folder")}</span>
                                  </button>
                                </div>,
                                document.body,
                              )
                            : null}
                        </div>
                      )}
                      {!collapsedGroup ? (
                        group.sessions.length > 0 ? (
                          <div className="pl-5">
                            <OrganizedSessionList
                              sessions={group.sessions}
                              courses={courses}
                              activeSessionId={activeSessionId}
                              onSelect={(sessionId) => {
                                drawer?.close();
                                return onSelectSession(sessionId);
                              }}
                              onRename={onRenameSession}
                              onDelete={onDeleteSession}
                              onOrganize={onOrganizeSession}
                              folderOptions={folders}
                              onMove={handleMoveSessionToFolder}
                            />
                          </div>
                        ) : (
                          <p className="px-7 py-1 text-[10px] text-[var(--muted-foreground)]/50">
                            {t("No conversations in this folder")}
                          </p>
                        )
                      ) : null}
                    </section>
                  );
                })}
              </div>
            ) : null}

            <div className="mt-4 px-2 py-1 text-[11.5px] font-normal text-[var(--muted-foreground)]/60">
              {t("Recents")}
            </div>
            {loadingSessions ? (
              <SessionList
                sessions={[]}
                activeSessionId={activeSessionId}
                loading
                onSelect={onSelectSession}
                onRename={onRenameSession}
                onDelete={onDeleteSession}
                compact
              />
            ) : onOrganizeSession && folderManagementEnabled ? (
              <OrganizedSessionList
                sessions={recentSessions}
                courses={courses}
                activeSessionId={activeSessionId}
                onSelect={(sessionId) => {
                  drawer?.close();
                  return onSelectSession(sessionId);
                }}
                onRename={onRenameSession}
                onDelete={onDeleteSession}
                onOrganize={onOrganizeSession}
                folderOptions={folders}
                onMove={handleMoveSessionToFolder}
              />
            ) : onOrganizeSession ? (
              <OrganizedSessionList
                sessions={sidebarSessions.slice(0, 8)}
                courses={courses}
                activeSessionId={activeSessionId}
                onSelect={(sessionId) => {
                  drawer?.close();
                  return onSelectSession(sessionId);
                }}
                onRename={onRenameSession}
                onDelete={onDeleteSession}
                onOrganize={onOrganizeSession}
              />
            ) : (
              <SessionList
                sessions={sidebarSessions.slice(0, 8)}
                activeSessionId={activeSessionId}
                onSelect={(sessionId) => {
                  drawer?.close();
                  return onSelectSession(sessionId);
                }}
                onRename={onRenameSession}
                onDelete={onDeleteSession}
                compact
              />
            )}
          </div>
        </section>
      ) : null}

      {/* When chat history is unavailable, fill the gap above the footer. */}
      {(!showSessions ||
        !onSelectSession ||
        !onRenameSession ||
        !onDeleteSession) && <div className="flex-1" />}

      {/* Secondary nav + footer */}
      <div className="border-t border-[var(--border)]/40 px-2 py-2">
        {SECONDARY_NAV.map((item) => {
          const active = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={closeDrawerOnNav}
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] transition-colors ${
                active
                  ? "bg-[var(--accent)] font-medium text-[var(--foreground)]"
                  : "text-[var(--foreground)]/85 hover:bg-[var(--background)]/60 hover:text-[var(--foreground)]"
              }`}
            >
              <item.icon size={16} strokeWidth={active ? 1.9 : 1.5} />
              <span>{t(item.label)}</span>
            </Link>
          );
        })}
        {renderedFooter}
        <div className="mt-0.5 flex items-center gap-0.5">
          <VersionBadge />
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noreferrer noopener"
            title={t("Docs") as string}
            aria-label={t("Docs") as string}
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-[var(--muted-foreground)]/55 transition-colors hover:bg-[var(--background)]/50 hover:text-[var(--muted-foreground)]"
          >
            <BookText size={13} strokeWidth={1.7} />
          </a>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            title="GitHub"
            aria-label="GitHub"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-[var(--muted-foreground)]/55 transition-colors hover:bg-[var(--background)]/50 hover:text-[var(--muted-foreground)]"
          >
            <Github size={13} strokeWidth={1.7} />
          </a>
        </div>
      </div>
    </aside>
  );
}
