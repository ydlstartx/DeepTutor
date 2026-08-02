import test from "node:test";
import assert from "node:assert/strict";
import katex from "katex";
import {
  convertFlowFenceToMermaid,
  convertLatexDelimiters,
  convertSequenceFenceToMermaid,
  makeLatexCompatibilityRemarkPlugin,
  normalizeLatexCjkInMath,
  normalizeLatexTextMiddleDots,
  processLatexContent,
  processMarkdownContent,
  resolveKatexStrictMode,
} from "../lib/latex";

// ---------------------------------------------------------------------------
// convertLatexDelimiters
// ---------------------------------------------------------------------------

test("convertLatexDelimiters: \\\\[...\\\\] → block $$...$$", () => {
  const input = "Before \\[x^2 + 1\\] after";
  const result = convertLatexDelimiters(input);
  assert.ok(result.includes("$$\nx^2 + 1\n$$"));
});

test("convertLatexDelimiters: \\\\(...\\\\) → inline $...$", () => {
  const input = "Solve \\(x = 2\\) now";
  const result = convertLatexDelimiters(input);
  assert.ok(result.includes("$x = 2$"));
});

test("convertLatexDelimiters: strips \\\\(\\\\) inside $$...$$", () => {
  const input = "$$\\(x^2\\)$$";
  const result = convertLatexDelimiters(input);
  assert.ok(result.includes("$$\nx^2\n$$"));
  assert.ok(!result.includes("\\("));
});

test("convertLatexDelimiters: multiline \\\\[...\\\\]", () => {
  const input = "\\[\n\\frac{a}{b}\n\\]";
  const result = convertLatexDelimiters(input);
  assert.ok(result.includes("$$"));
  assert.ok(result.includes("\\frac{a}{b}"));
});

test("convertLatexDelimiters: preserves expr containing $& replacement char", () => {
  const input = "\\[x \\$\\& y\\]";
  const result = convertLatexDelimiters(input);
  assert.ok(
    result.includes("x \\$\\& y"),
    "special regex replacement char $& must be preserved",
  );
});

test("convertLatexDelimiters: returns empty-ish input unchanged", () => {
  assert.equal(convertLatexDelimiters(""), "");
  assert.equal(convertLatexDelimiters(null as unknown as string), null);
});

test("convertLatexDelimiters: collapses triple+ newlines", () => {
  const input = "a\n\n\n\nb";
  const result = convertLatexDelimiters(input);
  assert.ok(!result.includes("\n\n\n"));
});

// ---------------------------------------------------------------------------
// processLatexContent (thin wrapper)
// ---------------------------------------------------------------------------

test("processLatexContent: delegates to convertLatexDelimiters", () => {
  assert.equal(processLatexContent(""), "");
  const out = processLatexContent("\\(x\\)");
  assert.ok(out.includes("$x$"));
});

test("SI units: moves Unicode middle dot out of LaTeX text mode", () => {
  const input = "$1\\,\\text{N} = 1\\,\\text{kg·m/s}^2$";
  const expected =
    "$1\\,\\text{N} = 1\\,\\text{kg}\\mathbin{\\cdot}\\text{m/s}^2$";
  assert.equal(normalizeLatexTextMiddleDots(input), expected);
  assert.equal(processMarkdownContent(input), expected);
  const html = katex.renderToString(expected.slice(1, -1), {
    output: "html",
    throwOnError: false,
  });
  assert.ok(!html.includes("color:#cc0000"));
  assert.ok(!html.includes("\\cdotp"));
});

test("SI units: leaves middle dots in prose and code unchanged", () => {
  const input = "正文 A·B，代码 `\\text{kg·m}`。";
  assert.equal(normalizeLatexTextMiddleDots(input), input);
});

test("CJK labels: moves raw Chinese into LaTeX text mode", () => {
  const input = "F_{合}+F_弹+k=斜率+b=截距";
  const expected =
    "F_{\\text{合}}+F_{\\text{弹}}+k=\\text{斜率}+b=\\text{截距}";
  assert.equal(normalizeLatexCjkInMath(input), expected);

  const strictEvents: string[] = [];
  katex.renderToString(expected, {
    throwOnError: true,
    strict: (errorCode: string) => {
      strictEvents.push(errorCode);
      return "ignore";
    },
  });
  assert.deepEqual(strictEvents, []);
});

test("CJK labels: preserves existing text-mode commands", () => {
  const input =
    "F_{\\text{合}}+\\operatorname{斜率}(x)+\\mathrm{N}+\\text{弹力}";
  assert.equal(normalizeLatexCjkInMath(input), input);
});

test("KaTeX strict mode: ignores only Unicode text in math mode", () => {
  assert.equal(resolveKatexStrictMode("unicodeTextInMathMode"), "ignore");
  assert.equal(resolveKatexStrictMode("unknownSymbol"), "warn");
  assert.equal(resolveKatexStrictMode("htmlExtension"), "warn");
});

test("CJK labels: remark plugin changes math nodes only", () => {
  const plugin = makeLatexCompatibilityRemarkPlugin();
  const tree = {
    type: "root",
    children: [
      { type: "text", value: "正文 F_{合}" },
      { type: "inlineCode", value: "$F_{合}$" },
      { type: "inlineMath", value: "F_{合}" },
      { type: "math", value: "F_{弹}=kx" },
    ],
  };

  plugin()(tree);
  assert.equal(tree.children[0].value, "正文 F_{合}");
  assert.equal(tree.children[1].value, "$F_{合}$");
  assert.equal(tree.children[2].value, "F_{\\text{合}}");
  assert.equal(tree.children[3].value, "F_{\\text{弹}}=kx");
});

test("CJK labels: remark plugin updates remark-math rendering payload", () => {
  const plugin = makeLatexCompatibilityRemarkPlugin();
  const tree = {
    type: "root",
    children: [
      {
        type: "inlineMath",
        value: "F_合=ma+F_弹",
        data: {
          hChildren: [{ type: "text", value: "F_合=ma+F_弹" }],
        },
      },
    ],
  };

  plugin()(tree);
  const renderedValue = tree.children[0].data.hChildren[0].value;
  assert.equal(renderedValue, "F_{\\text{合}}=ma+F_{\\text{弹}}");

  const strictEvents: string[] = [];
  katex.renderToString(renderedValue, {
    throwOnError: true,
    strict: (errorCode: string) => {
      strictEvents.push(errorCode);
      return "ignore";
    },
  });
  assert.deepEqual(strictEvents, []);
});

// ---------------------------------------------------------------------------
// processMarkdownContent — heading normalisation
// ---------------------------------------------------------------------------

test("headings: inserts space when missing", () => {
  const result = processMarkdownContent("##Title");
  assert.ok(result.includes("## Title"));
});

test("headings: leaves properly-spaced headings alone", () => {
  const result = processMarkdownContent("## Title");
  assert.ok(result.includes("## Title"));
});

test("headings: does not split 7+ consecutive hashes", () => {
  const result = processMarkdownContent("#######");
  assert.equal(result.trim(), "#######");
});

// ---------------------------------------------------------------------------
// processMarkdownContent — inline math normalisation ($$...$$ → $...$)
// ---------------------------------------------------------------------------

test("inline math: $$x$$ on a line with other text → $x$", () => {
  const result = processMarkdownContent("Compute $$x^2$$ now");
  assert.ok(result.includes("$x^2$"));
  assert.ok(!result.includes("$$x^2$$"));
});

test("inline math: standalone one-line $$...$$ → split to block", () => {
  const result = processMarkdownContent("$$\\frac{a}{b}$$");
  const lines = result.trim().split("\n");
  assert.equal(lines[0], "$$");
  assert.ok(lines[1].includes("\\frac{a}{b}"));
  assert.equal(lines[2], "$$");
});

// ---------------------------------------------------------------------------
// processMarkdownContent — loose block math ($...\n...\n$) promotion
// ---------------------------------------------------------------------------

test("loose block math: single-$ lines wrapping LaTeX → promoted to $$", () => {
  const input = "$\n\\frac{a}{b}\n$";
  const result = processMarkdownContent(input);
  const lines = result.trim().split("\n");
  assert.equal(lines[0], "$$");
  assert.ok(lines.some((l) => l.includes("\\frac{a}{b}")));
  assert.equal(lines[lines.length - 1], "$$");
});

test("loose block math: single-$ lines with non-LaTeX content → not promoted", () => {
  const input = "$\nhello world\n$";
  const result = processMarkdownContent(input);
  assert.ok(!result.startsWith("$$"));
});

test("loose block math: recognises \\\\ (line break) as LaTeX", () => {
  const input = "$\na \\\\\nb\n$";
  const result = processMarkdownContent(input);
  const lines = result.trim().split("\n");
  assert.equal(lines[0], "$$");
});

test("loose block math: recognises _ and ^ as LaTeX", () => {
  const input = "$\nx_1 + y^2\n$";
  const result = processMarkdownContent(input);
  const lines = result.trim().split("\n");
  assert.equal(lines[0], "$$");
});

// ---------------------------------------------------------------------------
// processMarkdownContent — end-to-end pipeline
// ---------------------------------------------------------------------------

test("pipeline: combined heading + inline math + delimiter conversion", () => {
  const input = "##Heading\n\nSolve \\(x=1\\) and $$y$$";
  const result = processMarkdownContent(input);
  assert.ok(result.includes("## Heading"), "heading should be normalised");
  assert.ok(result.includes("$x=1$"), "\\\\(...\\\\) should become $...$");
  assert.ok(result.includes("$y$"), "$$y$$ inline should become $y$");
});

test("pipeline: preserves fenced code blocks", () => {
  const input = "```python\nprint('hello')\n```";
  const result = processMarkdownContent(input);
  assert.ok(result.includes("print('hello')"));
});

// ---------------------------------------------------------------------------
// editor.md fence conversion (flow / seq)
// ---------------------------------------------------------------------------

test("flow fence: keeps yes/no branch labels from the source side", () => {
  const input = [
    "st=>start: Start",
    "cond=>condition: Ready?",
    "a=>operation: Go",
    "b=>operation: Wait",
    "e=>end: Done",
    "st->cond",
    "cond(yes)->a->e",
    "cond(no)->b->e",
  ].join("\n");
  const result = convertFlowFenceToMermaid(input);
  assert.ok(result, "conversion should succeed");
  assert.ok(result.includes("cond -->|yes| a"));
  assert.ok(result.includes("cond -->|no| b"));
});

test("flow fence: layout hints are not edge labels", () => {
  const input = [
    "st=>start: Start",
    "op=>operation: Work",
    "e=>end: Done",
    "st(right)->op->e",
  ].join("\n");
  const result = convertFlowFenceToMermaid(input);
  assert.ok(result, "conversion should succeed");
  assert.ok(result.includes("st --> op"));
  assert.ok(!result.includes("|right|"));
});

test("seq fence: converts messages and notes", () => {
  const input = [
    "Student->DeepTutor: Ask for help",
    "Note right of DeepTutor: Collect memory\\nand context",
    "DeepTutor-->Student: Respond",
  ].join("\n");
  const result = convertSequenceFenceToMermaid(input);
  assert.ok(result, "conversion should succeed");
  assert.ok(result.startsWith("sequenceDiagram"));
  assert.ok(result.includes("Student->>DeepTutor: Ask for help"));
  assert.ok(result.includes("Collect memory<br/>and context"));
});
