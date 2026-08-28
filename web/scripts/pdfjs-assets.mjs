#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const webRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);

const defaultSourceDir = path.join(
  webRoot,
  "node_modules",
  "pdfjs-dist",
  "cmaps",
);
const defaultTargetDir = path.join(webRoot, "public", "pdfjs", "cmaps");

/**
 * Copy pdf.js' packed Adobe CMaps into the public tree.
 *
 * pdf.js needs these files at runtime for Identity-H/V CID fonts. They remain
 * generated rather than checked in so their version always matches the
 * installed pdfjs-dist package. Both dev.mjs and build.mjs call this before
 * Next starts; production images and Python wheels already package public/.
 */
export function preparePdfjsCMaps({
  sourceDir = defaultSourceDir,
  targetDir = defaultTargetDir,
} = {}) {
  if (!existsSync(sourceDir)) {
    throw new Error(
      `pdf.js CMaps are missing at ${sourceDir}; install web dependencies first.`,
    );
  }

  mkdirSync(targetDir, { recursive: true });
  cpSync(sourceDir, targetDir, { recursive: true, force: true });

  return readdirSync(targetDir).filter((name) => name.endsWith(".bcmap")).length;
}

const isEntry =
  import.meta.url === pathToFileURL(process.argv[1] ?? "").href;

if (isEntry) {
  const count = preparePdfjsCMaps();
  console.log(`Prepared ${count} packed PDF.js CMaps.`);
}
