/**
 * Case-insensitive username filter backing the admin Users search box.
 * An empty / whitespace-only query returns the input list unchanged.
 *
 * Generic over `{ username }` (rather than importing UserRecord) so the
 * module stays alias-free and loadable by the node unit tests.
 */
export function filterUsersByQuery<T extends { username: string }>(
  users: T[],
  query: string,
): T[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return users;
  return users.filter((user) =>
    user.username.toLowerCase().includes(normalized),
  );
}

export type UserActivityFilter = "all" | "active_7d" | "inactive_30d";
export type UserActivitySort = "last_used" | "username";

type ActivityFields = {
  username: string;
  last_used_at?: string | null;
};

function activityTimestamp(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function filterUsersByActivity<T extends ActivityFields>(
  users: T[],
  filter: UserActivityFilter,
  now = Date.now(),
): T[] {
  if (filter === "all") return users;
  const cutoff = now - (filter === "active_7d" ? 7 : 30) * 24 * 60 * 60 * 1000;
  if (filter === "active_7d") {
    return users.filter((user) => activityTimestamp(user.last_used_at) >= cutoff);
  }
  return users.filter((user) => activityTimestamp(user.last_used_at) < cutoff);
}

export function sortUsersByActivity<T extends ActivityFields>(
  users: T[],
  sort: UserActivitySort,
): T[] {
  return [...users].sort((left, right) => {
    if (sort === "username") {
      return left.username.localeCompare(right.username, undefined, {
        sensitivity: "base",
      });
    }
    const byActivity =
      activityTimestamp(right.last_used_at) - activityTimestamp(left.last_used_at);
    return byActivity || left.username.localeCompare(right.username);
  });
}
