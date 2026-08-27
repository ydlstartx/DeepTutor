"use client";

import { Fragment, useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { fetchAuthStatus } from "@/lib/auth";
import {
  listUserActivity,
  deleteUser,
  setUserRole,
  createUser,
  resetUserPassword,
  type UserRecord,
  type UserActivityRecord,
  type UserActivityStatus,
  type UserUsageSummary,
} from "@/lib/admin-api";
import { GrantEditor } from "@/features/multi-user/components/GrantEditor";
import { UserAvatar } from "@/components/UserAvatar";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import {
  filterUsersByActivity,
  filterUsersByQuery,
  sortUsersByActivity,
  type UserActivityFilter,
  type UserActivitySort,
} from "@/lib/admin-users";
import {
  Activity,
  Search,
  Shield,
  ShieldCheck,
  ShieldOff,
  Trash2,
  RefreshCw,
  ArrowLeft,
  SlidersHorizontal,
  UserPlus,
  Users,
  KeyRound,
  ChevronDown,
  Clock3,
  MessageSquareText,
  X,
} from "lucide-react";
import Link from "next/link";
import { formatDate as formatLocaleDate, type Language } from "@/lib/datetime";

// Delegates to the shared locale mapping so a new UI language only has to be
// taught to lib/datetime; the guard here is for the empty or unparseable
// created_at that Intl would throw on.
function formatDate(iso: string, lang: Language): string {
  if (!iso) return "—";
  try {
    return formatLocaleDate(new Date(iso), lang);
  } catch {
    return "—";
  }
}

function formatOptionalDate(iso: string | null, lang: Language): string {
  if (!iso) return "—";
  try {
    return formatLocaleDate(new Date(iso), lang, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

function formatNumber(value: number, lang: Language): string {
  return new Intl.NumberFormat(lang === "zh" ? "zh-CN" : "en-US").format(
    value,
  );
}

const STATUS_STYLES: Record<UserActivityStatus, string> = {
  recent: "bg-emerald-500",
  today: "bg-sky-500",
  recent_7d: "bg-amber-500",
  inactive: "bg-[var(--muted-foreground)]/45",
};

function UsagePeriod({
  label,
  usage,
  lang,
}: {
  label: string;
  usage: UserUsageSummary;
  lang: Language;
}) {
  const { t } = useTranslation();
  const metrics = [
    [t("Conversations"), usage.conversations],
    [t("Completed turns"), usage.completed_turns],
    [t("Failed turns"), usage.failed_turns],
    [t("Knowledge base queries"), usage.kb_queries],
    [t("LLM calls"), usage.llm_calls],
    [t("Tokens"), usage.total_tokens],
  ] as const;
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h4 className="text-sm font-medium text-[var(--foreground)]">{label}</h4>
        {!usage.usage_complete && (
          <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
            {t("Partial usage data")}
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-5 gap-y-3 sm:grid-cols-3">
        {metrics.map(([name, value]) => (
          <div key={name}>
            <p className="text-[11px] text-[var(--muted-foreground)]">{name}</p>
            <p className="mt-0.5 text-sm font-semibold tabular-nums text-[var(--foreground)]">
              {formatNumber(value, lang)}
            </p>
          </div>
        ))}
      </div>
      <div className="mt-3 border-t border-[var(--border)] pt-3 text-xs text-[var(--muted-foreground)]">
        {t("Estimated cost")}: {`$${usage.estimated_cost_usd.toFixed(6)}`}
      </div>
    </div>
  );
}

function UserActivityDetails({
  user,
  lang,
}: {
  user: UserActivityRecord;
  lang: Language;
}) {
  const { t } = useTranslation();
  return (
    <div className="bg-[var(--background)]/45 px-5 py-4">
      <div className="mb-4 grid gap-3 text-xs text-[var(--muted-foreground)] sm:grid-cols-3">
        <div>
          <p>{t("Last login")}</p>
          <p className="mt-1 font-medium text-[var(--foreground)]">
            {formatOptionalDate(user.last_login_at, lang)}
          </p>
        </div>
        <div>
          <p>{t("Last online")}</p>
          <p className="mt-1 font-medium text-[var(--foreground)]">
            {formatOptionalDate(user.last_seen_at, lang)}
          </p>
        </div>
        <div>
          <p>{t("Joined")}</p>
          <p className="mt-1 font-medium text-[var(--foreground)]">
            {formatDate(user.created_at, lang)}
          </p>
        </div>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <UsagePeriod label={t("Last 7 days")} usage={user.usage_7d} lang={lang} />
        <UsagePeriod label={t("Last 30 days")} usage={user.usage_30d} lang={lang} />
      </div>
      <p className="mt-3 text-[11px] text-[var(--muted-foreground)]">
        {t("Usage totals contain metadata only. Conversation content is never shown here.")}
      </p>
    </div>
  );
}

export default function AdminUsersPage() {
  const router = useRouter();
  const { t, i18n } = useTranslation();
  const lang: Language = i18n.language?.startsWith("zh") ? "zh" : "en";
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [users, setUsers] = useState<UserActivityRecord[]>([]);
  const [summary, setSummary] = useState({
    total_users: 0,
    active_today: 0,
    active_7d: 0,
    inactive_30d: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [actionNotice, setActionNotice] = useState("");
  const [expandedUserId, setExpandedUserId] = useState<string | null>(null);
  const [activityExpandedUserId, setActivityExpandedUserId] = useState<
    string | null
  >(null);
  const [activityFilter, setActivityFilter] =
    useState<UserActivityFilter>("all");
  const [activitySort, setActivitySort] =
    useState<UserActivitySort>("last_used");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [query, setQuery] = useState("");
  const [confirmTarget, setConfirmTarget] = useState<{
    kind: "delete" | "promote" | "demote";
    user: UserRecord;
  } | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const [createUsername, setCreateUsername] = useState("");
  const [createPassword, setCreatePassword] = useState("");
  const [createSubmitting, setCreateSubmitting] = useState(false);
  const [createError, setCreateError] = useState("");
  const [resetTarget, setResetTarget] = useState<UserRecord | null>(null);
  const [resetPassword, setResetPassword] = useState("");
  const [resetConfirmation, setResetConfirmation] = useState("");
  const [resetSubmitting, setResetSubmitting] = useState(false);
  const [resetError, setResetError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    setActionNotice("");
    try {
      const data = await listUserActivity();
      setUsers(data.users);
      setSummary(data.summary);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("Failed to load users"));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchAuthStatus().then((status) => {
      if (!status?.authenticated) {
        router.replace("/login");
        return;
      }
      if (status.role !== "admin") {
        router.replace("/");
        return;
      }
      setCurrentUser(status.username ?? null);
      void load();
    });
  }, [router, load]);

  function openCreateDialog() {
    setCreateUsername("");
    setCreatePassword("");
    setCreateError("");
    setShowCreateDialog(true);
  }

  function closeCreateDialog() {
    if (createSubmitting) return;
    setShowCreateDialog(false);
  }

  async function handleCreateSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (createSubmitting) return;
    setCreateError("");
    const username = createUsername.trim();
    if (!username) {
      setCreateError(t("Username is required."));
      return;
    }
    if (createPassword.length < 8) {
      setCreateError(t("Password must be at least 8 characters."));
      return;
    }
    setCreateSubmitting(true);
    try {
      await createUser(username, createPassword);
      setShowCreateDialog(false);
      await load();
    } catch (e) {
      setCreateError(
        e instanceof Error ? e.message : t("Failed to create user"),
      );
    } finally {
      setCreateSubmitting(false);
    }
  }

  function openResetDialog(user: UserRecord) {
    setResetTarget(user);
    setResetPassword("");
    setResetConfirmation("");
    setResetError("");
  }

  function closeResetDialog() {
    if (resetSubmitting) return;
    setResetTarget(null);
  }

  async function handleResetSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!resetTarget || resetSubmitting) return;
    setResetError("");
    if (resetPassword.length < 8) {
      setResetError(t("Password must be at least 8 characters."));
      return;
    }
    if (resetPassword !== resetConfirmation) {
      setResetError(t("Passwords do not match"));
      return;
    }
    setResetSubmitting(true);
    try {
      await resetUserPassword(resetTarget.username, resetPassword);
      setActionNotice(
        t("Password reset for {{username}}.", {
          username: resetTarget.username,
        }),
      );
      setResetTarget(null);
    } catch (e) {
      setResetError(
        e instanceof Error ? t(e.message) : t("Failed to reset password"),
      );
    } finally {
      setResetSubmitting(false);
    }
  }

  async function handleConfirmAction() {
    if (!confirmTarget || confirmBusy) return;
    const { kind, user } = confirmTarget;
    setConfirmBusy(true);
    setActionError("");
    setActionNotice("");
    try {
      if (kind === "delete") {
        await deleteUser(user.username);
      } else {
        const newRole = kind === "promote" ? "admin" : "user";
        await setUserRole(user.username, newRole);
        if (newRole === "admin") {
          setExpandedUserId((current) =>
            current === user.id ? null : current,
          );
        }
      }
      await load();
      setConfirmTarget(null);
    } catch (e) {
      setConfirmTarget(null);
      setActionError(
        e instanceof Error
          ? e.message
          : confirmTarget.kind === "delete"
            ? t("Failed to delete user")
            : t("Failed to update role"),
      );
    } finally {
      setConfirmBusy(false);
    }
  }

  useEffect(() => {
    if (!expandedUserId) return;
    const expanded = users.find((user) => user.id === expandedUserId);
    if (!expanded || expanded.role === "admin") {
      setExpandedUserId(null);
    }
  }, [expandedUserId, users]);

  useEffect(() => {
    if (
      activityExpandedUserId &&
      !users.some((user) => user.id === activityExpandedUserId)
    ) {
      setActivityExpandedUserId(null);
    }
  }, [activityExpandedUserId, users]);

  const normalizedQuery = query.trim().toLowerCase();
  const filteredUsers = sortUsersByActivity(
    filterUsersByActivity(filterUsersByQuery(users, query), activityFilter),
    activitySort,
  );

  return (
    <div className="h-screen overflow-y-auto bg-[var(--background)] px-4 py-10 [scrollbar-gutter:stable]">
      <div className="mx-auto max-w-6xl">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="mb-4 inline-flex items-center gap-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
          >
            <ArrowLeft size={16} />
            {t("Back")}
          </Link>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="font-serif text-xl font-semibold text-[var(--foreground)]">
                {t("User Management")}
              </h1>
              <p className="mt-0.5 text-sm text-[var(--muted-foreground)]">
                {t("Manage accounts, assignments, and privacy-safe activity")}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button
                onClick={openCreateDialog}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm
                           border border-[var(--border)] text-[var(--foreground)]
                           hover:bg-[var(--card)] transition-colors"
              >
                <UserPlus size={14} />
                {t("Add user")}
              </button>
              <button
                onClick={load}
                disabled={loading}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm
                           border border-[var(--border)] text-[var(--muted-foreground)]
                           hover:text-[var(--foreground)] hover:bg-[var(--card)]
                           disabled:opacity-50 transition-colors"
              >
                <RefreshCw
                  size={14}
                  className={loading ? "animate-spin" : ""}
                />
                {t("Refresh")}
              </button>
            </div>
          </div>
        </div>

        {actionError && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600 dark:text-red-400">
            {actionError}
          </div>
        )}
        {actionNotice && (
          <div className="mb-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-400">
            {actionNotice}
          </div>
        )}

        {!loading && !error && (
          <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              {
                label: t("Total users"),
                value: summary.total_users,
                icon: Users,
              },
              {
                label: t("Active today"),
                value: summary.active_today,
                icon: Activity,
              },
              {
                label: t("Active in 7 days"),
                value: summary.active_7d,
                icon: MessageSquareText,
              },
              {
                label: t("Inactive for 30 days"),
                value: summary.inactive_30d,
                icon: Clock3,
              },
            ].map((card) => (
              <div
                key={card.label}
                className="rounded-xl border border-[var(--border)] bg-[var(--card)] px-4 py-3 shadow-sm"
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="text-xs text-[var(--muted-foreground)]">{card.label}</p>
                  <card.icon size={15} className="text-[var(--muted-foreground)]" />
                </div>
                <p className="mt-2 text-2xl font-semibold tabular-nums text-[var(--foreground)]">
                  {formatNumber(card.value, lang)}
                </p>
              </div>
            ))}
          </div>
        )}

        {!loading && !error && users.length > 0 && (
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <div className="relative flex-1">
              <Search
                size={14}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
              />
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t("Search users…")}
                aria-label={t("Search users")}
                className="w-full rounded-lg border border-[var(--border)] bg-[var(--card)] py-2 pl-9 pr-3 text-sm
                           text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]/70
                           outline-none focus:border-[var(--ring)] transition-colors"
              />
            </div>
            <select
              value={activityFilter}
              onChange={(event) =>
                setActivityFilter(event.target.value as UserActivityFilter)
              }
              aria-label={t("Filter by activity")}
              className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
            >
              <option value="all">{t("All activity")}</option>
              <option value="active_7d">{t("Active in 7 days")}</option>
              <option value="inactive_30d">{t("Inactive for 30 days")}</option>
            </select>
            <select
              value={activitySort}
              onChange={(event) =>
                setActivitySort(event.target.value as UserActivitySort)
              }
              aria-label={t("Sort users")}
              className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
            >
              <option value="last_used">{t("Recently used")}</option>
              <option value="username">{t("Username")}</option>
            </select>
            <span className="shrink-0 text-xs text-[var(--muted-foreground)]">
              {normalizedQuery || activityFilter !== "all"
                ? t("{{filtered}} of {{total}}", {
                    filtered: filteredUsers.length,
                    total: users.length,
                  })
                : t(users.length === 1 ? "{{count}} user" : "{{count}} users", {
                    count: users.length,
                  })}
            </span>
          </div>
        )}

        <div className="rounded-2xl border border-[var(--border)] bg-[var(--card)] overflow-x-auto shadow-sm">
          {loading ? (
            <div className="divide-y divide-[var(--border)]" aria-hidden>
              {[0, 1, 2].map((row) => (
                <div
                  key={row}
                  className="flex animate-pulse items-center gap-3 px-5 py-4"
                >
                  <div className="h-8 w-8 rounded-full bg-[var(--muted)]/60" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3 w-36 rounded bg-[var(--muted)]/60" />
                    <div className="h-2.5 w-24 rounded bg-[var(--muted)]/40" />
                  </div>
                  <div className="h-5 w-16 rounded-full bg-[var(--muted)]/40" />
                </div>
              ))}
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-16 text-red-500 text-sm">
              {error}
            </div>
          ) : users.length === 0 ? (
            <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
              <Users
                size={28}
                strokeWidth={1.5}
                className="text-[var(--muted-foreground)]/50"
              />
              <p className="mt-3 text-sm font-medium text-[var(--foreground)]">
                {t("No users yet")}
              </p>
              <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                {t("Accounts you create will appear here.")}
              </p>
              <button
                onClick={openCreateDialog}
                className="mt-4 flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm
                           border border-[var(--border)] text-[var(--foreground)]
                           hover:bg-[var(--background)]/60 transition-colors"
              >
                <UserPlus size={14} />
                {t("Add user")}
              </button>
            </div>
          ) : filteredUsers.length === 0 ? (
            <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
              <Search
                size={28}
                strokeWidth={1.5}
                className="text-[var(--muted-foreground)]/50"
              />
              <p className="mt-3 text-sm font-medium text-[var(--foreground)]">
                {normalizedQuery
                  ? t("No users match “{{query}}”", { query: query.trim() })
                  : t("No users match the selected activity filter")}
              </p>
              <button
                onClick={() => {
                  setQuery("");
                  setActivityFilter("all");
                }}
                className="mt-4 rounded-lg px-3 py-1.5 text-sm border border-[var(--border)]
                           text-[var(--muted-foreground)] hover:text-[var(--foreground)]
                           hover:bg-[var(--background)]/60 transition-colors"
              >
                {normalizedQuery ? t("Clear search") : t("Clear filters")}
              </button>
            </div>
          ) : (
            <table className="w-full min-w-[1050px] text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-left text-xs text-[var(--muted-foreground)] uppercase tracking-wider">
                  <th className="px-5 py-3 font-medium">{t("Username")}</th>
                  <th className="px-5 py-3 font-medium">{t("Role")}</th>
                  <th className="px-5 py-3 font-medium">{t("Activity")}</th>
                  <th className="px-5 py-3 font-medium">{t("Last login")}</th>
                  <th className="px-5 py-3 font-medium">{t("Last used")}</th>
                  <th className="px-5 py-3 font-medium">{t("Last 7 days")}</th>
                  <th className="px-5 py-3 font-medium text-right">
                    {t("Actions")}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)]">
                {filteredUsers.map((user) => {
                  const isSelf = user.username === currentUser;
                  const isAdmin = user.role === "admin";
                  const canManageAssignments = !isAdmin && Boolean(user.id);
                  const activityLabel =
                    user.activity_status === "recent"
                      ? t("Just active")
                      : user.activity_status === "today"
                        ? t("Active today")
                        : user.activity_status === "recent_7d"
                          ? t("Active in 7 days")
                          : t("Inactive");
                  return (
                    <Fragment key={user.username}>
                      <tr className="group hover:bg-[var(--background)]/50 transition-colors">
                        <td className="px-5 py-3">
                          <div className="flex items-center gap-3">
                            <UserAvatar
                              username={user.username}
                              userId={user.id}
                              avatar={user.avatar}
                              role={user.role}
                              size={32}
                            />
                            <span className="min-w-0 truncate font-medium text-[var(--foreground)]">
                              {user.username}
                              {isSelf && (
                                <span className="ml-2 text-xs font-normal text-[var(--muted-foreground)]">
                                  {t("(you)")}
                                </span>
                              )}
                            </span>
                          </div>
                        </td>
                        <td className="px-5 py-3">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium
                            ${
                              isAdmin
                                ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                                : "bg-[var(--muted)]/50 text-[var(--muted-foreground)]"
                            }`}
                          >
                            {isAdmin && (
                              <ShieldCheck size={11} strokeWidth={2} />
                            )}
                            {isAdmin ? t("Admin") : t("User")}
                          </span>
                        </td>
                        <td className="px-5 py-3.5">
                          <button
                            onClick={() =>
                              setActivityExpandedUserId((current) =>
                                current === user.id ? null : user.id,
                              )
                            }
                            title={t("View activity details")}
                            className="inline-flex items-center gap-2 rounded-lg px-2 py-1 text-xs text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)]"
                          >
                            <span
                              className={`h-2 w-2 rounded-full ${STATUS_STYLES[user.activity_status]}`}
                            />
                            {activityLabel}
                            <ChevronDown
                              size={13}
                              className={`transition-transform ${
                                activityExpandedUserId === user.id ? "rotate-180" : ""
                              }`}
                            />
                          </button>
                        </td>
                        <td className="whitespace-nowrap px-5 py-3.5 text-xs text-[var(--muted-foreground)]">
                          {formatOptionalDate(user.last_login_at, lang)}
                        </td>
                        <td className="whitespace-nowrap px-5 py-3.5 text-xs text-[var(--muted-foreground)]">
                          {formatOptionalDate(user.last_used_at, lang)}
                        </td>
                        <td className="whitespace-nowrap px-5 py-3.5 text-xs text-[var(--muted-foreground)]">
                          {t("{{turns}} turns · {{queries}} KB", {
                            turns: formatNumber(user.usage_7d.turns, lang),
                            queries: formatNumber(user.usage_7d.kb_queries, lang),
                          })}
                        </td>
                        <td className="px-5 py-3.5">
                          <div className="flex items-center justify-end gap-1.5">
                            {canManageAssignments && (
                              <button
                                onClick={() =>
                                  setExpandedUserId((current) =>
                                    current === user.id ? null : user.id,
                                  )
                                }
                                title={t("Manage assignments")}
                                className="rounded-lg p-1.5 text-[var(--muted-foreground)]
                                         hover:bg-[var(--background)] hover:text-[var(--foreground)]
                                         transition-colors"
                              >
                                <SlidersHorizontal size={15} />
                              </button>
                            )}
                            <button
                              onClick={() => openResetDialog(user)}
                              disabled={isSelf}
                              title={
                                isSelf
                                  ? t("Change your own password from your profile")
                                  : t("Reset password for {{username}}", {
                                      username: user.username,
                                    })
                              }
                              aria-label={
                                isSelf
                                  ? t("Change your own password from your profile")
                                  : t("Reset password for {{username}}", {
                                      username: user.username,
                                    })
                              }
                              className="rounded-lg p-1.5 text-[var(--muted-foreground)]
                                       hover:bg-[var(--background)] hover:text-[var(--foreground)]
                                       disabled:cursor-not-allowed disabled:opacity-30 transition-colors"
                            >
                              <KeyRound size={15} />
                            </button>
                            <button
                              onClick={() =>
                                setConfirmTarget({
                                  kind: isAdmin ? "demote" : "promote",
                                  user,
                                })
                              }
                              disabled={isSelf}
                              title={
                                isSelf
                                  ? t("Cannot change your own role")
                                  : user.role === "admin"
                                    ? t("Demote to user")
                                    : t("Promote to admin")
                              }
                              className="rounded-lg p-1.5 text-[var(--muted-foreground)]
                                       hover:bg-[var(--background)] hover:text-[var(--foreground)]
                                       disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                              {user.role === "admin" ? (
                                <ShieldOff size={15} />
                              ) : (
                                <Shield size={15} />
                              )}
                            </button>
                            <button
                              onClick={() =>
                                setConfirmTarget({ kind: "delete", user })
                              }
                              disabled={isSelf}
                              title={
                                isSelf
                                  ? t("Cannot delete your own account")
                                  : t("Delete {{username}}", {
                                      username: user.username,
                                    })
                              }
                              className="rounded-lg p-1.5 text-[var(--muted-foreground)]
                                       hover:bg-red-500/10 hover:text-red-500
                                       disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                            >
                              <Trash2 size={15} />
                            </button>
                          </div>
                        </td>
                      </tr>
                      {activityExpandedUserId === user.id && (
                        <tr>
                          <td colSpan={7} className="p-0">
                            <UserActivityDetails user={user} lang={lang} />
                          </td>
                        </tr>
                      )}
                      {canManageAssignments && expandedUserId === user.id && (
                        <tr>
                          <td colSpan={7} className="p-0">
                            <GrantEditor key={user.id} userId={user.id} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        <p className="mt-8 text-center text-xs text-[var(--muted-foreground)]">
          {t("DeepTutor Admin · User Management")}
        </p>
      </div>

      <ConfirmDialog
        open={confirmTarget !== null}
        title={
          confirmTarget?.kind === "delete"
            ? t("Delete user")
            : confirmTarget?.kind === "promote"
              ? t("Promote to admin")
              : t("Demote to user")
        }
        tone={confirmTarget?.kind === "delete" ? "danger" : "default"}
        confirmLabel={
          confirmTarget?.kind === "delete"
            ? t("Delete user")
            : confirmTarget?.kind === "promote"
              ? t("Promote")
              : t("Demote")
        }
        busyLabel={
          confirmTarget?.kind === "delete"
            ? t("Deleting…")
            : confirmTarget?.kind === "promote"
              ? t("Promoting…")
              : t("Demoting…")
        }
        busy={confirmBusy}
        onConfirm={handleConfirmAction}
        onCancel={() => setConfirmTarget(null)}
      >
        {confirmTarget && (
          <>
            <div className="flex items-center gap-3 rounded-xl border border-[var(--border)] bg-[var(--background)]/50 px-3 py-2.5">
              <UserAvatar
                username={confirmTarget.user.username}
                userId={confirmTarget.user.id}
                avatar={confirmTarget.user.avatar}
                role={confirmTarget.user.role}
                size={32}
              />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-[var(--foreground)]">
                  {confirmTarget.user.username}
                </p>
                <p className="text-xs text-[var(--muted-foreground)]">
                  {t("{{role}} · joined {{date}}", {
                    role:
                      confirmTarget.user.role === "admin"
                        ? t("Admin")
                        : t("User"),
                    date: formatDate(confirmTarget.user.created_at, lang),
                  })}
                </p>
              </div>
            </div>
            <p className="mt-3">
              {confirmTarget.kind === "delete"
                ? t(
                    "This permanently removes the account and its assignments. This cannot be undone.",
                  )
                : confirmTarget.kind === "promote"
                  ? t(
                      "Admins can manage users and assignments, and work in the shared main workspace.",
                    )
                  : t(
                      "They will lose access to the admin area and switch to their own assigned workspace.",
                    )}
            </p>
          </>
        )}
      </ConfirmDialog>

      {showCreateDialog && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--overlay)] px-4"
          role="dialog"
          aria-modal="true"
          onClick={closeCreateDialog}
        >
          <form
            onClick={(e) => e.stopPropagation()}
            onSubmit={handleCreateSubmit}
            className="w-full max-w-sm rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-xl"
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-base font-semibold text-[var(--foreground)]">
                {t("Add user")}
              </h2>
              <button
                type="button"
                onClick={closeCreateDialog}
                disabled={createSubmitting}
                className="rounded-md p-1 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)] disabled:opacity-40"
                aria-label={t("Close")}
              >
                <X size={16} />
              </button>
            </div>

            <label className="mb-3 block text-xs text-[var(--muted-foreground)]">
              {t("Username (or email)")}
              <input
                type="text"
                value={createUsername}
                onChange={(e) => setCreateUsername(e.target.value)}
                disabled={createSubmitting}
                autoComplete="off"
                autoFocus
                className="mt-1 w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
              />
            </label>

            <label className="mb-4 block text-xs text-[var(--muted-foreground)]">
              {t("Password (≥ 8 chars)")}
              <input
                type="password"
                value={createPassword}
                onChange={(e) => setCreatePassword(e.target.value)}
                disabled={createSubmitting}
                autoComplete="new-password"
                className="mt-1 w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
              />
            </label>

            {createError && (
              <p className="mb-3 text-xs text-red-500">{createError}</p>
            )}

            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={closeCreateDialog}
                disabled={createSubmitting}
                className="rounded-lg px-3 py-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] disabled:opacity-40"
              >
                {t("Cancel")}
              </button>
              <button
                type="submit"
                disabled={createSubmitting}
                className="rounded-lg bg-[var(--foreground)] px-3 py-1.5 text-sm font-medium text-[var(--background)] hover:opacity-90 disabled:opacity-40"
              >
                {createSubmitting ? t("Creating…") : t("Create")}
              </button>
            </div>
          </form>
        </div>
      )}

      {resetTarget && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--overlay)] px-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="reset-password-title"
          onClick={closeResetDialog}
        >
          <form
            onClick={(e) => e.stopPropagation()}
            onSubmit={handleResetSubmit}
            className="w-full max-w-sm rounded-2xl border border-[var(--border)] bg-[var(--card)] p-5 shadow-xl"
          >
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2
                  id="reset-password-title"
                  className="text-base font-semibold text-[var(--foreground)]"
                >
                  {t("Reset password")}
                </h2>
                <p className="mt-0.5 text-xs text-[var(--muted-foreground)]">
                  {t("Set a new password for {{username}}", {
                    username: resetTarget.username,
                  })}
                </p>
              </div>
              <button
                type="button"
                onClick={closeResetDialog}
                disabled={resetSubmitting}
                className="rounded-md p-1 text-[var(--muted-foreground)] hover:bg-[var(--background)] hover:text-[var(--foreground)] disabled:opacity-40"
                aria-label={t("Close")}
              >
                <X size={16} />
              </button>
            </div>

            <label className="mb-3 block text-xs text-[var(--muted-foreground)]">
              {t("New password")}
              <input
                type="password"
                value={resetPassword}
                onChange={(e) => setResetPassword(e.target.value)}
                disabled={resetSubmitting}
                autoComplete="new-password"
                minLength={8}
                required
                autoFocus
                className="mt-1 w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
              />
            </label>

            <label className="mb-4 block text-xs text-[var(--muted-foreground)]">
              {t("Confirm new password")}
              <input
                type="password"
                value={resetConfirmation}
                onChange={(e) => setResetConfirmation(e.target.value)}
                disabled={resetSubmitting}
                autoComplete="new-password"
                minLength={8}
                required
                className="mt-1 w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm text-[var(--foreground)] outline-none focus:border-[var(--ring)]"
              />
            </label>

            {resetError && (
              <p className="mb-3 text-xs text-red-500">{resetError}</p>
            )}

            <p className="mb-4 text-xs text-[var(--muted-foreground)]">
              {t("This signs the user out of their existing sessions.")}
            </p>

            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={closeResetDialog}
                disabled={resetSubmitting}
                className="rounded-lg px-3 py-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] disabled:opacity-40"
              >
                {t("Cancel")}
              </button>
              <button
                type="submit"
                disabled={resetSubmitting}
                className="rounded-lg bg-[var(--foreground)] px-3 py-1.5 text-sm font-medium text-[var(--background)] hover:opacity-90 disabled:opacity-40"
              >
                {resetSubmitting ? t("Resetting…") : t("Reset password")}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
