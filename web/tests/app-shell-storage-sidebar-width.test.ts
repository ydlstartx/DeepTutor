import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  normalizeSidebarWidth,
} from "../context/app-shell-storage";

test("sidebar width uses a stable default", () => {
  assert.equal(normalizeSidebarWidth(null), DEFAULT_SIDEBAR_WIDTH);
  assert.equal(normalizeSidebarWidth("not-a-number"), DEFAULT_SIDEBAR_WIDTH);
});

test("sidebar width is rounded and clamped", () => {
  assert.equal(normalizeSidebarWidth(247.6), 248);
  assert.equal(normalizeSidebarWidth(100), MIN_SIDEBAR_WIDTH);
  assert.equal(normalizeSidebarWidth(900), MAX_SIDEBAR_WIDTH);
});
