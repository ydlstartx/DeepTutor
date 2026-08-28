import { apiFetch, apiUrl } from "@/lib/api";

export interface UserRecord {
  id: string;
  username: string;
  role: "admin" | "user";
  created_at: string;
  disabled?: boolean;
  /** Avatar marker: "", "icon:<name>:<color>", or "img:<version>". */
  avatar?: string;
  book_permission?: {
    create: boolean;
    default: "none" | "read";
    books: Record<string, "none" | "read" | "edit">;
  };
}

export type UserActivityStatus =
  | "recent"
  | "today"
  | "recent_7d"
  | "inactive";

export interface UserUsageSummary {
  conversations: number;
  turns: number;
  completed_turns: number;
  failed_turns: number;
  running_turns: number;
  cancelled_turns: number;
  kb_queries: number;
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  usage_complete: boolean;
  usage_reported_turns: number;
  history_complete: boolean;
}

export interface UserActivityRecord extends UserRecord {
  activity_status: UserActivityStatus;
  last_activity_at: string | null;
  last_login_at: string | null;
  last_seen_at: string | null;
  last_used_at: string | null;
  usage_7d: UserUsageSummary;
  usage_30d: UserUsageSummary;
}

export interface UserActivityReport {
  summary: {
    total_users: number;
    active_today: number;
    active_7d: number;
    inactive_30d: number;
  };
  users: UserActivityRecord[];
  generated_at: string | null;
  retention_days: number;
}

export async function listUsers(): Promise<UserRecord[]> {
  const res = await apiFetch(apiUrl("/api/v1/auth/users"));
  if (!res.ok) throw new Error("Failed to fetch users");
  return res.json();
}

export async function listUserActivity(): Promise<UserActivityReport> {
  const res = await apiFetch(apiUrl("/api/v1/auth/users/activity"));
  if (!res.ok) throw new Error("Failed to fetch user activity");
  return res.json();
}

export async function deleteUser(username: string): Promise<void> {
  const res = await apiFetch(
    apiUrl(`/api/v1/auth/users/${encodeURIComponent(username)}`),
    {
      method: "DELETE",
    },
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail ?? "Failed to delete user");
  }
}

export async function setUserRole(
  username: string,
  role: "admin" | "user",
): Promise<void> {
  const res = await apiFetch(
    apiUrl(`/api/v1/auth/users/${encodeURIComponent(username)}/role`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    },
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail ?? "Failed to update role");
  }
}

export async function resetUserPassword(
  username: string,
  newPassword: string,
): Promise<void> {
  const res = await apiFetch(
    apiUrl(`/api/v1/auth/users/${encodeURIComponent(username)}/password`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_password: newPassword }),
    },
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail ?? "Failed to reset password");
  }
}

export interface CreatedUser {
  user_id: string;
  username: string;
  role: "admin" | "user";
  is_admin: boolean;
}

export async function createUser(
  username: string,
  password: string,
): Promise<CreatedUser> {
  const res = await apiFetch(apiUrl("/api/v1/auth/users"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail) && detail.length > 0 && detail[0]?.msg
          ? String(detail[0].msg)
          : "Failed to create user";
    throw new Error(message);
  }
  return (await res.json()) as CreatedUser;
}
