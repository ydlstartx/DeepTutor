import assert from "node:assert/strict";
import test from "node:test";

import {
  importExistingKnowledgeBase,
  listKnowledgeImportFolders,
  probeKnowledgeImportFolder,
} from "../lib/knowledge-api";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(
  handler: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>,
): () => void {
  const original = globalThis.fetch;
  (globalThis as { fetch: typeof fetch }).fetch = handler;
  return () => {
    (globalThis as { fetch: typeof fetch }).fetch = original;
  };
}

test("knowledge import API uses only upload-relative paths", async () => {
  const calls: Array<{ input: string; body: unknown }> = [];
  const restore = stubFetch(async (input, init) => {
    calls.push({
      input: String(input),
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    if (String(input).includes("/folders")) {
      return jsonResponse(200, {
        path: "courses",
        parent: "",
        candidate: false,
        folders: [],
      });
    }
    if (String(input).endsWith("/probe")) {
      return jsonResponse(200, {
        ok: true,
        path: "courses/high-school",
        suggested_name: "High school",
        provider: "llamaindex",
        version_count: 1,
        ready_version_count: 1,
        document_count: 10,
        file_count: 5,
        size_bytes: 1024,
        warnings: [],
        error: null,
      });
    }
    return jsonResponse(200, {
      status: "imported",
      name: "High school",
      source_path: "courses/high-school",
      rag_provider: "llamaindex",
      file_count: 5,
      size_bytes: 1024,
    });
  });

  try {
    await listKnowledgeImportFolders("courses");
    await probeKnowledgeImportFolder("courses/high-school");
    const result = await importExistingKnowledgeBase({
      path: "courses/high-school",
      name: "High school",
    });

    assert.equal(calls[0].input, "/api/v1/knowledge/import/folders?path=courses");
    assert.deepEqual(calls[1].body, { path: "courses/high-school" });
    assert.deepEqual(calls[2].body, {
      path: "courses/high-school",
      name: "High school",
    });
    assert.equal(result.status, "imported");
  } finally {
    restore();
  }
});
