import assert from "node:assert/strict";
import test from "node:test";

import { resetUserPassword } from "../lib/admin-api";
import { changePassword } from "../lib/profile-api";

test("password APIs send credentials only to their scoped endpoints", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; method: string; body: unknown }> = [];
  globalThis.fetch = async (input, init) => {
    calls.push({
      url: String(input),
      method: String(init?.method),
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    await changePassword("current-secret", "new-secret-1234");
    await resetUserPassword("student@example.com", "reset-secret-1234");

    assert.deepEqual(calls, [
      {
        url: "/api/v1/auth/profile/password",
        method: "PUT",
        body: {
          current_password: "current-secret",
          new_password: "new-secret-1234",
        },
      },
      {
        url: "/api/v1/auth/users/student%40example.com/password",
        method: "PUT",
        body: { new_password: "reset-secret-1234" },
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
