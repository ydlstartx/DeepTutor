import test from "node:test";
import assert from "node:assert/strict";

import {
  collectReferencedImageUrls,
  makeFileLinkRemarkPlugin,
} from "../components/common/InlineFileCard";

const GENERATED_IMAGE = {
  type: "image",
  filename: "newton1_vt.png",
  mime_type: "image/png",
  url: "/api/outputs/workspace/chat/chat/turn_1/exec/newton1_vt.png",
  generated: true,
} as const;

test("exact generated filename in inline code becomes an attachment link", () => {
  const plugin = makeFileLinkRemarkPlugin([GENERATED_IMAGE]);
  assert.ok(plugin);
  const tree: Record<string, unknown> = {
    type: "root",
    children: [
      {
        type: "paragraph",
        children: [
          { type: "text", value: "配图 " },
          { type: "inlineCode", value: "newton1_vt.png" },
        ],
      },
    ],
  };

  plugin()(tree);
  const paragraph = (tree.children as Array<Record<string, unknown>>)[0];
  const children = paragraph.children as Array<Record<string, unknown>>;
  assert.equal(children[1].type, "link");
  assert.equal(children[1].url, "attachment:newton1_vt.png");
});

test("unrelated inline code remains inline code", () => {
  const plugin = makeFileLinkRemarkPlugin([GENERATED_IMAGE]);
  assert.ok(plugin);
  const tree: Record<string, unknown> = {
    type: "root",
    children: [
      {
        type: "paragraph",
        children: [{ type: "inlineCode", value: "velocity = distance / time" }],
      },
    ],
  };

  plugin()(tree);
  const paragraph = (tree.children as Array<Record<string, unknown>>)[0];
  const children = paragraph.children as Array<Record<string, unknown>>;
  assert.equal(children[0].type, "inlineCode");
});

test("a generated image is auto-expanded only at its first filename mention", () => {
  const plugin = makeFileLinkRemarkPlugin([GENERATED_IMAGE]);
  assert.ok(plugin);
  const tree: Record<string, unknown> = {
    type: "root",
    children: [
      {
        type: "paragraph",
        children: [{ type: "inlineCode", value: "newton1_vt.png" }],
      },
      {
        type: "paragraph",
        children: [{ type: "inlineCode", value: "newton1_vt.png" }],
      },
    ],
  };

  plugin()(tree);
  const paragraphs = tree.children as Array<Record<string, unknown>>;
  const first = paragraphs[0].children as Array<Record<string, unknown>>;
  const second = paragraphs[1].children as Array<Record<string, unknown>>;
  assert.equal(first[0].type, "link");
  assert.equal(second[0].type, "inlineCode");
});

test("backticked generated filename still counts as rendered in the body", () => {
  const urls = collectReferencedImageUrls(
    "**配图** `newton1_vt.png`：速度保持不变。",
    [GENERATED_IMAGE],
  );
  assert.deepEqual([...urls], [GENERATED_IMAGE.url]);
});
