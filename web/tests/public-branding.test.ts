import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

const readWebFile = (...parts: string[]) =>
  readFileSync(path.resolve(process.cwd(), ...parts), "utf8");

test("public branding constants match the deployed site", () => {
  const source = readWebFile("lib", "public-brand.ts");

  assert.match(source, /PUBLIC_PRODUCT_NAME = "知序求索"/);
  assert.match(source, /PUBLIC_SITE_URL = "https:\/\/tutor\.oppgent\.com"/);
  assert.match(source, /ICP_RECORD_NUMBER = "粤ICP备2026117324号"/);
  assert.match(source, /ICP_RECORD_URL = "https:\/\/beian\.miit\.gov\.cn\/"/);
  assert.match(
    source,
    /UPSTREAM_PROJECT_URL = "https:\/\/github\.com\/HKUDS\/DeepTutor"/,
  );
  assert.match(
    source,
    /APACHE_LICENSE_URL =\s*"https:\/\/www\.apache\.org\/licenses\/LICENSE-2\.0"/,
  );
  assert.match(source, /PUBLIC_FOOTER_NON_OFFICIAL = "非 DeepTutor 官方服务"/);
});

test("public footer exposes filing and upstream attribution as safe external links", () => {
  const source = readWebFile(
    "components",
    "layout",
    "PublicSiteFooter.tsx",
  );

  assert.match(source, /ICP_RECORD_NUMBER/);
  assert.match(source, /UPSTREAM_PROJECT_NAME/);
  assert.match(source, /APACHE_LICENSE_NAME/);
  assert.match(source, /PUBLIC_FOOTER_NON_OFFICIAL/);
  assert.equal((source.match(/target="_blank"/g) ?? []).length, 3);
  assert.equal((source.match(/rel="noopener noreferrer"/g) ?? []).length, 3);
});

test("public footer is mounted across workspace, auth, and admin shells", () => {
  for (const file of [
    ["components", "layout", "AppShell.tsx"],
    ["app", "(auth)", "layout.tsx"],
    ["app", "(admin)", "layout.tsx"],
  ]) {
    assert.match(readWebFile(...file), /<PublicSiteFooter \/>/);
  }
});

test("browser metadata and primary shell use the public brand", () => {
  const metadata = readWebFile("app", "layout.tsx");
  const shell = readWebFile("components", "layout", "AppShell.tsx");
  const sidebar = readWebFile("components", "sidebar", "SidebarShell.tsx");

  assert.match(metadata, /title: PUBLIC_PRODUCT_NAME/);
  assert.match(metadata, /description: PUBLIC_PRODUCT_DESCRIPTION/);
  assert.match(metadata, /url: "\/public-brand\.svg"/);

  for (const source of [shell, sidebar]) {
    assert.match(source, /PUBLIC_PRODUCT_NAME/);
    assert.doesNotMatch(source, /\/(?:logo|banner)-sm\.png/);
  }
});
