import test from "node:test";
import assert from "node:assert/strict";

import {
  filterUsersByActivity,
  filterUsersByQuery,
  sortUsersByActivity,
} from "../lib/admin-users";

function user(username: string): { id: string; username: string } {
  return { id: `u_${username}`, username };
}

const USERS = [user("Alice"), user("bob"), user("alice.smith"), user("Карина")];

test("empty or whitespace-only query returns the list unchanged", () => {
  assert.equal(filterUsersByQuery(USERS, ""), USERS);
  assert.equal(filterUsersByQuery(USERS, "   "), USERS);
});

test("matching is case-insensitive and substring-based", () => {
  assert.deepEqual(
    filterUsersByQuery(USERS, "ALICE").map((u) => u.username),
    ["Alice", "alice.smith"],
  );
  assert.deepEqual(
    filterUsersByQuery(USERS, "smith").map((u) => u.username),
    ["alice.smith"],
  );
});

test("query is trimmed before matching", () => {
  assert.deepEqual(
    filterUsersByQuery(USERS, "  bob  ").map((u) => u.username),
    ["bob"],
  );
});

test("non-latin usernames are searchable", () => {
  assert.deepEqual(
    filterUsersByQuery(USERS, "кари").map((u) => u.username),
    ["Карина"],
  );
});

test("no match yields an empty list, not an error", () => {
  assert.deepEqual(filterUsersByQuery(USERS, "zzz"), []);
  assert.deepEqual(filterUsersByQuery([], "alice"), []);
});

test("activity filters use the unified last-activity time", () => {
  const now = Date.parse("2026-08-27T12:00:00Z");
  const users = [
    {
      username: "today",
      last_activity_at: "2026-08-27T11:00:00Z",
      last_used_at: "2026-08-27T11:00:00Z",
    },
    {
      username: "seen-only",
      last_activity_at: "2026-08-27T10:00:00Z",
      last_used_at: null,
    },
    {
      username: "week",
      last_activity_at: "2026-08-22T12:00:00Z",
      last_used_at: "2026-08-22T12:00:00Z",
    },
    {
      username: "old",
      last_activity_at: "2026-07-01T12:00:00Z",
      last_used_at: "2026-07-01T12:00:00Z",
    },
    { username: "never", last_activity_at: null, last_used_at: null },
  ];

  assert.deepEqual(
    filterUsersByActivity(users, "active_7d", now).map((u) => u.username),
    ["today", "seen-only", "week"],
  );
  assert.deepEqual(
    filterUsersByActivity(users, "inactive_30d", now).map((u) => u.username),
    ["old", "never"],
  );
});

test("activity sort keeps users without usage last", () => {
  const users = [
    { username: "never", last_used_at: null },
    { username: "older", last_used_at: "2026-08-20T00:00:00Z" },
    { username: "latest", last_used_at: "2026-08-27T00:00:00Z" },
  ];
  assert.deepEqual(
    sortUsersByActivity(users, "last_used").map((u) => u.username),
    ["latest", "older", "never"],
  );
});
