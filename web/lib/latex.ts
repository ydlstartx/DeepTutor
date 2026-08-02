/**
 * Utility functions for LaTeX processing
 *
 * remark-math only supports $...$ and $$...$$ delimiters by default.
 * Many LLMs output LaTeX using \(...\) and \[...\] delimiters.
 * This utility converts between formats.
 */

const MATH_SEGMENT_RE =
  /\$\$[\s\S]*?\$\$|\$(?!\$)[^$\n]*?\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)/g;

const CJK_CHARACTER_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
const TEXT_MODE_COMMANDS = new Set([
  "emph",
  "mbox",
  "operatorname",
  "text",
  "textbf",
  "textit",
  "textnormal",
  "textrm",
  "textsf",
  "texttt",
]);

/**
 * KaTeX supports CJK glyphs through the browser font stack but reports each
 * raw glyph as a strict-mode compatibility warning.  The remark plugin below
 * normally moves those labels into `\\text{...}`; this focused fallback keeps
 * intentionally supported CJK labels from flooding the browser console if a
 * third-party markdown transform bypasses or overwrites that normalization.
 * Other strict-mode diagnostics remain visible.
 */
export function resolveKatexStrictMode(errorCode: string): "ignore" | "warn" {
  return errorCode === "unicodeTextInMathMode" ? "ignore" : "warn";
}

type MarkdownAstNode = {
  type?: string;
  value?: unknown;
  children?: MarkdownAstNode[];
  data?: {
    hChildren?: MarkdownAstNode[];
    [key: string]: unknown;
  };
};

function balancedGroupEnd(value: string, start: number): number {
  if (value[start] !== "{") return -1;
  let depth = 0;
  for (let index = start; index < value.length; index += 1) {
    const char = value[index];
    if (char === "\\") {
      index += 1;
      continue;
    }
    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

/**
 * Move raw CJK labels into LaTeX text mode while preserving existing text
 * commands.  KaTeX renders `F_{合}` but reports `unicodeTextInMathMode`;
 * `F_{\text{合}}` is both visually correct and LaTeX-compatible.
 */
export function normalizeLatexCjkInMath(expression: string): string {
  if (!expression || !CJK_CHARACTER_RE.test(expression)) return expression;

  let result = "";
  let index = 0;
  while (index < expression.length) {
    const char = expression[index];
    if (char === "\\") {
      const commandMatch = /^\\([A-Za-z]+)/.exec(expression.slice(index));
      if (!commandMatch) {
        result += expression.slice(index, index + 2);
        index += 2;
        continue;
      }

      const command = commandMatch[1];
      const commandEnd = index + commandMatch[0].length;
      if (TEXT_MODE_COMMANDS.has(command)) {
        let groupStart = commandEnd;
        if (expression[groupStart] === "*") groupStart += 1;
        while (/\s/.test(expression[groupStart] ?? "")) groupStart += 1;
        const groupEnd = balancedGroupEnd(expression, groupStart);
        if (groupEnd >= 0) {
          result += expression.slice(index, groupEnd + 1);
          index = groupEnd + 1;
          continue;
        }
      }

      result += commandMatch[0];
      index = commandEnd;
      continue;
    }

    if (CJK_CHARACTER_RE.test(char)) {
      let end = index + 1;
      while (end < expression.length && CJK_CHARACTER_RE.test(expression[end])) {
        end += 1;
      }
      const text = `\\text{${expression.slice(index, end)}}`;
      result += ["_", "^"].includes(expression[index - 1]) ? `{${text}}` : text;
      index = end;
      continue;
    }

    result += char;
    index += 1;
  }
  return result;
}

/**
 * remark plugin applied after remark-math.  Working on `math`/`inlineMath`
 * nodes avoids rewriting prose, inline code, or fenced code examples.
 */
export function makeLatexCompatibilityRemarkPlugin() {
  const visit = (node: MarkdownAstNode): void => {
    if (
      (node.type === "math" || node.type === "inlineMath") &&
      typeof node.value === "string"
    ) {
      const normalizedValue = normalizeLatexCjkInMath(
        normalizeLatexTextMiddleDotsInMath(node.value),
      );
      node.value = normalizedValue;

      // remark-math stores a second copy of the expression for the mdast →
      // hast bridge.  Updating only `node.value` looks correct in the remark
      // tree, but rehype-katex still receives the stale `hChildren` value.
      // Keep that rendering payload in sync so KaTeX sees the normalized TeX.
      const renderedMathText = node.data?.hChildren;
      if (Array.isArray(renderedMathText) && renderedMathText.length === 1) {
        const textNode = renderedMathText[0];
        if (textNode.type === "text" && typeof textNode.value === "string") {
          textNode.value = normalizedValue;
        }
      }
    }
    for (const child of node.children ?? []) visit(child);
  };
  return () => (tree: MarkdownAstNode) => visit(tree);
}

function normalizeLatexTextMiddleDotsInMath(expression: string): string {
  return expression.replace(
    /\\text\{([^{}]*[·⋅][^{}]*)\}/g,
    (_match, inner: string) => {
      const parts = inner.split(/[·⋅]/);
      return parts.map((part) => `\\text{${part}}`).join("\\mathbin{\\cdot}");
    },
  );
}

/**
 * KaTeX treats a Unicode middle dot inside `\text{...}` as the unsupported
 * text command `\cdotp`, which is rendered as red error text. Keep the prose
 * on either side in text mode and move the multiplication operator back into
 * math mode: `\text{kg·m}` -> `\text{kg}\mathbin{\cdot}\text{m}`.
 *
 * Only math-delimited segments are touched, so code/prose examples containing
 * `\text{...}` remain byte-for-byte unchanged.
 */
export function normalizeLatexTextMiddleDots(content: string): string {
  if (!content) return content;
  return content.replace(MATH_SEGMENT_RE, (math) =>
    normalizeLatexTextMiddleDotsInMath(math),
  );
}

/**
 * Convert LaTeX delimiters from \(...\) and \[...\] to $...$ and $$...$$
 * This makes the content compatible with remark-math for ReactMarkdown rendering.
 *
 * @param content - The content containing LaTeX with \(...\) or \[...\] delimiters
 * @returns Content with $...$ and $$...$$ delimiters
 */
export function convertLatexDelimiters(content: string): string {
  if (!content) return content;

  let result = content;

  // editor.md examples sometimes wrap \( ... \) inside $$ ... $$.
  // In that case the inner delimiters should be stripped rather than rewrapped.
  result = result.replace(
    /\$\$\s*\\\(([\s\S]*?)\\\)\s*\$\$/g,
    (_match, expr) => {
      return `\n$$\n${expr}\n$$\n`;
    },
  );

  // Convert \[...\] to $$...$$ (block math).
  // Use a regex that handles multiline content
  // Note: In JSON strings, \[ becomes \\[ which in JS becomes \[
  result = result.replace(/\\\[([\s\S]*?)\\\]/g, (_match, expr) => {
    return `\n$$\n${expr}\n$$\n`;
  });

  // Convert \(...\) to $...$ (inline math).
  // Be careful not to match escaped parentheses in other contexts
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, (_match, expr) => {
    return ` $${expr}$ `;
  });

  // Also handle cases where LaTeX is directly in the text without proper delimiters
  // e.g., standalone \lim, \frac, etc. that should be wrapped
  // This is a common issue with LLM outputs

  // Clean up multiple consecutive newlines
  result = result.replace(/\n{3,}/g, "\n\n");

  return result;
}

function normalizeEditorMdHeadings(content: string): string {
  return content.replace(/^(#{1,6})([^#\s])/gm, "$1 $2");
}

const LIKELY_LATEX_BLOCK_RE = /\\[A-Za-z]+|\\\\|[_^&]/;

function looksLikeLatexBlock(lines: string[]): boolean {
  const block = lines
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n");

  return block.length > 0 && LIKELY_LATEX_BLOCK_RE.test(block);
}

function normalizeEditorMdInlineMath(content: string): string {
  const lines = content.split("\n");
  const result: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed === "$" && i + 1 < lines.length) {
      let endIdx = -1;
      for (let j = i + 1; j < lines.length; j++) {
        if (lines[j].trim() === "$") {
          endIdx = j;
          break;
        }
      }

      if (endIdx > i + 1 && looksLikeLatexBlock(lines.slice(i + 1, endIdx))) {
        result.push("$$");
        for (let j = i + 1; j < endIdx; j++) {
          result.push(lines[j]);
        }
        result.push("$$");
        i = endIdx;
        continue;
      }
    }

    if (
      /^\$\$[\s\S]+\$\$$/.test(trimmed) &&
      (trimmed.match(/\$\$/g)?.length ?? 0) === 2
    ) {
      const inner = trimmed.slice(2, -2).trim();
      result.push(`$$\n${inner}\n$$`);
      continue;
    }

    // editor.md commonly uses $...$ for inline math.
    result.push(
      line.replace(/\$\$([^$\n]+?)\$\$/g, (_match, expr: string) => {
        return `$${expr.trim()}$`;
      }),
    );
  }

  return result.join("\n");
}

type HeadingEntry = {
  level: number;
  text: string;
  slug: string;
};

function slugifyHeading(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
}

function collectMarkdownHeadings(content: string): HeadingEntry[] {
  const lines = content.split("\n");
  const headings: HeadingEntry[] = [];
  let inFence = false;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (/^```/.test(trimmed)) {
      inFence = !inFence;
      continue;
    }

    if (inFence) continue;

    const atxMatch = /^(#{1,6})\s+(.+?)\s*$/.exec(line);
    if (atxMatch) {
      const text = atxMatch[2].replace(/\s+#+\s*$/, "").trim();
      const slug = slugifyHeading(text);
      if (slug) headings.push({ level: atxMatch[1].length, text, slug });
      continue;
    }

    const next = lines[i + 1]?.trim();
    if (!trimmed || !next) continue;

    if (/^=+$/.test(next)) {
      const slug = slugifyHeading(trimmed);
      if (slug) headings.push({ level: 1, text: trimmed, slug });
      i += 1;
      continue;
    }

    if (/^-+$/.test(next)) {
      const slug = slugifyHeading(trimmed);
      if (slug) headings.push({ level: 2, text: trimmed, slug });
      i += 1;
    }
  }

  return headings;
}

function buildTableOfContents(headings: HeadingEntry[]): string {
  if (headings.length === 0) return "";

  return headings
    .map(({ level, text, slug }) => {
      const indent = "  ".repeat(Math.max(level - 1, 0));
      return `${indent}- [${text}](#${slug})`;
    })
    .join("\n");
}

function injectEditorMdTableOfContents(content: string): string {
  const headings = collectMarkdownHeadings(content);
  if (headings.length === 0) {
    return content.replace(/^\[TOCM?\]\s*$/gim, "");
  }

  const toc = buildTableOfContents(headings);
  return content.replace(/^\[TOCM?\]\s*$/gim, toc);
}

export function convertFlowFenceToMermaid(source: string): string | null {
  const lines = source
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) return null;

  const nodeDefs: string[] = [];
  const edges: string[] = [];

  const renderNode = (id: string, type: string, label: string): string => {
    const safeLabel = label.replace(/\|/g, "\\|");
    switch (type) {
      case "start":
      case "end":
        return `${id}([${safeLabel}])`;
      case "condition":
        return `${id}{${safeLabel}}`;
      case "inputoutput":
        return `${id}[/${safeLabel}/]`;
      case "subroutine":
        return `${id}[[${safeLabel}]]`;
      case "database":
        return `${id}[(${safeLabel})]`;
      default:
        return `${id}[${safeLabel}]`;
    }
  };

  for (const line of lines) {
    const defMatch =
      /^([A-Za-z][\w-]*)(?:=>|=)(start|end|operation|condition|inputoutput|subroutine|database):\s*(.+)$/.exec(
        line,
      );

    if (defMatch) {
      const [, id, type, label] = defMatch;
      nodeDefs.push(`  ${renderNode(id, type, label)}`);
      continue;
    }

    if (!line.includes("->")) continue;

    const parts = line
      .split("->")
      .map((part) => part.trim())
      .filter(Boolean);
    for (let i = 0; i < parts.length - 1; i += 1) {
      const fromMatch = /^([A-Za-z][\w-]*)(?:\(([^)]+)\))?$/.exec(parts[i]);
      const toMatch = /^([A-Za-z][\w-]*)(?:\(([^)]+)\))?$/.exec(parts[i + 1]);
      if (!fromMatch || !toMatch) continue;

      const [, fromId, fromAnnotation] = fromMatch;
      const [, toId] = toMatch;
      // flowchart.js puts branch labels on the source side (`cond(yes)->x`);
      // pure layout hints (`op(right)->x`) are not labels.
      const branch = fromAnnotation?.split(",")[0]?.trim();
      const label =
        branch && !/^(left|right|top|bottom)$/i.test(branch)
          ? `|${branch}|`
          : "";
      edges.push(`  ${fromId} -->${label} ${toId}`);
    }
  }

  if (nodeDefs.length === 0 || edges.length === 0) return null;
  return ["flowchart TD", ...nodeDefs, ...edges].join("\n");
}

export function convertSequenceFenceToMermaid(source: string): string | null {
  const lines = source
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) return null;

  const isSequenceDirective = (line: string): boolean => {
    return (
      /^Note\s+(left|right)\s+of\s+/i.test(line) ||
      /^participant\s+/i.test(line) ||
      /^(title|autonumber|activate|deactivate|loop|rect|opt|alt|par|critical|break|box|create|destroy)\b/i.test(
        line,
      ) ||
      /^([A-Za-z][\w.-]*)(?:-{1,2}>>?|--?>)([A-Za-z][\w.-]*)\s*:\s*.+$/.test(
        line,
      )
    );
  };

  const normalizedLines: string[] = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    if (/^Note\s+(left|right)\s+of\s+/i.test(line)) {
      let combined = line;

      while (i + 1 < lines.length && !isSequenceDirective(lines[i + 1])) {
        combined += `\\n${lines[i + 1]}`;
        i += 1;
      }

      normalizedLines.push(combined);
      continue;
    }

    normalizedLines.push(line);
  }

  const converted = normalizedLines.map((line) => {
    if (/^Note\s+(left|right)\s+of\s+/i.test(line)) {
      return `  ${line.replace(/\\n/g, "<br/>")}`;
    }

    const messageMatch =
      /^([A-Za-z][\w.-]*)(-{1,2}>>?|--?>)([A-Za-z][\w.-]*)\s*:\s*(.+)$/.exec(
        line,
      );

    if (messageMatch) {
      const [, from, operator, to, message] = messageMatch;
      const arrow =
        operator === "--" || operator === "-->"
          ? "-->>"
          : operator === "->>" || operator === "-->>"
            ? operator
            : "->>";
      return `  ${from}${arrow}${to}: ${message}`;
    }

    return `  ${line}`;
  });

  return ["sequenceDiagram", ...converted].join("\n");
}

function convertEditorMdFences(content: string): string {
  return content.replace(
    /```(flow|seq|sequence)\s*\n([\s\S]*?)```/g,
    (_match, lang: string, body: string) => {
      const converted =
        lang === "flow"
          ? convertFlowFenceToMermaid(body)
          : convertSequenceFenceToMermaid(body);

      if (!converted) return `\`\`\`${lang}\n${body}\`\`\``;
      return `\`\`\`mermaid\n${converted}\n\`\`\``;
    },
  );
}

/**
 * Process content for ReactMarkdown rendering with proper LaTeX support
 * This is a convenience wrapper that applies all necessary transformations.
 *
 * @param content - The raw content to process
 * @returns Processed content ready for ReactMarkdown with remark-math
 */
export function processLatexContent(content: string): string {
  if (!content) return "";

  // Convert to string if not already
  const str = String(content);

  // Apply delimiter conversion
  return convertLatexDelimiters(normalizeLatexTextMiddleDots(str));
}

export function processMarkdownContent(content: string): string {
  if (!content) return "";

  let result = String(content);
  result = normalizeLatexTextMiddleDots(result);
  result = normalizeEditorMdHeadings(result);
  result = normalizeEditorMdInlineMath(result);
  result = convertEditorMdFences(result);
  result = injectEditorMdTableOfContents(result);
  result = convertLatexDelimiters(result);
  result = result.replace(/\n{3,}/g, "\n\n");

  return result;
}
