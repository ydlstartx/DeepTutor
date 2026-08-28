import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const webRoot = process.cwd();
const read = (...parts: string[]) =>
  readFileSync(path.join(webRoot, ...parts), "utf8");

test("every PDF document load uses the shared packed CMap options", () => {
  const loader = read("lib", "pdfjs-loader.ts");
  assert.match(loader, /PDFJS_CMAP_URL\s*=\s*["']\/pdfjs\/cmaps\//);
  assert.match(loader, /cMapUrl:\s*PDFJS_CMAP_URL/);
  assert.match(loader, /cMapPacked:\s*true/);

  for (const file of [
    ["components", "chat", "preview", "previewers", "PdfPreview.tsx"],
    ["components", "reading", "PdfDocumentView.tsx"],
  ]) {
    assert.match(
      read(...file),
      /pdfDocumentInit\(/,
      `${file.join("/")} must use the shared CMap options`,
    );
  }
});

test("dev and production builds prepare matching pdfjs-dist CMaps", () => {
  for (const script of ["dev.mjs", "build.mjs"]) {
    assert.match(
      read("scripts", script),
      /preparePdfjsCMaps\(\)/,
      `${script} must prepare CMaps before starting Next`,
    );
  }
});

test("the asset preparer copies packed CMaps and their license", () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "deeptutor-pdfjs-cmaps-"));
  const sourceDir = path.join(root, "source");
  const targetDir = path.join(root, "target");
  mkdirSync(sourceDir);
  writeFileSync(path.join(sourceDir, "Adobe-GB1-UCS2.bcmap"), "fixture");
  writeFileSync(path.join(sourceDir, "LICENSE"), "license");

  try {
    const scriptUrl = pathToFileURL(
      path.join(webRoot, "scripts", "pdfjs-assets.mjs"),
    );
    const result = spawnSync(
      process.execPath,
      [
        "--input-type=module",
        "--eval",
        [
          `import { preparePdfjsCMaps } from ${JSON.stringify(scriptUrl.href)};`,
          `const count = preparePdfjsCMaps(${JSON.stringify({ sourceDir, targetDir })});`,
          `process.stdout.write(String(count));`,
        ].join("\n"),
      ],
      { encoding: "utf8" },
    );

    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, "1");
    assert.equal(
      readFileSync(path.join(targetDir, "Adobe-GB1-UCS2.bcmap"), "utf8"),
      "fixture",
    );
    assert.equal(readFileSync(path.join(targetDir, "LICENSE"), "utf8"), "license");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
