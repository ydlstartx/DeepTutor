import {
  PUBLIC_PRODUCT_NAME,
  PUBLIC_SITE_URL,
} from "@/lib/public-brand";

const FENCE = "```";

export const CO_WRITER_SAMPLE_TEMPLATE = `# ${PUBLIC_PRODUCT_NAME} Co-Writer

> ${PUBLIC_PRODUCT_NAME}的内置写作画布，适用于笔记、报告、教程和 AI 辅助草稿。

### Features

- Support Standard Markdown / CommonMark / GFM for everyday writing
- Real-time preview for headings, tables, code, math, flowchart, and sequence diagrams
- AI editing workflows for rewrite, shorten, and expand
- HTML tag decoding for tags like <sub>, <sup>, <abbr>, and <mark>
- A practical starter draft for ${PUBLIC_PRODUCT_NAME} product docs and learning content

## Headers (Underline)

${PUBLIC_PRODUCT_NAME} Learning Note
=============

${PUBLIC_PRODUCT_NAME} Study Outline
-------------

### Characters

----

~~Deprecated behavior~~ <s>Legacy formatting path</s>
*Italic* _Italic_
**Emphasis** __Emphasis__
***Emphasis Italic*** ___Emphasis Italic___

Superscript: X<sup>2</sup>, Subscript: O<sub>2</sub>

**Abbreviation(link HTML abbr tag)**

The <abbr title="Large Language Model">LLM</abbr> layer powers ${PUBLIC_PRODUCT_NAME} while the <abbr title="Retrieval Augmented Generation">RAG</abbr> layer provides grounded knowledge support.

### Blockquotes

> ${PUBLIC_PRODUCT_NAME} helps students turn questions into structured understanding.
>
> "Learn deeply, write clearly.", [${PUBLIC_PRODUCT_NAME}](#知序求索-co-writer)

### Links

[${PUBLIC_PRODUCT_NAME} Co-Writer](#知序求索-co-writer "co-writer section")

[${PUBLIC_PRODUCT_NAME} Learning Note](#知序求索-learning-note)

[${PUBLIC_PRODUCT_NAME} Website](${PUBLIC_SITE_URL})

[Reference link][deeptutor-doc]

[deeptutor-doc]: #deeptutor-learning-note

### Code Blocks

#### Inline code

\`deeptutor chat --once "Summarize this section"\`

#### Code Blocks (Indented style)

    from deeptutor.runtime.orchestrator import ChatOrchestrator
    orchestrator = ChatOrchestrator()
    print("${PUBLIC_PRODUCT_NAME} is ready.")

#### Python

${FENCE}python
from deeptutor.runtime.orchestrator import ChatOrchestrator
from deeptutor.core.context import UnifiedContext


async def run_demo() -> str:
    orchestrator = ChatOrchestrator()
    context = UnifiedContext(
        user_query="Explain Newton's second law",
        capability="chat",
    )
    result = await orchestrator.run(context)
    return result.get("response", "")
${FENCE}

#### JSON config

${FENCE}json
{
  "app_name": "${PUBLIC_PRODUCT_NAME}",
  "default_capability": "chat",
  "enabled_tools": ["rag", "web_search", "code_execution", "reason"],
  "ui": {
    "co_writer_template": true
  }
}
${FENCE}

#### HTML code

${FENCE}html
<section class="deeptutor-card">
  <h1>${PUBLIC_PRODUCT_NAME}</h1>
  <p>Write, revise, and organize learning content with AI.</p>
</section>
${FENCE}

### Images

![](/public-brand.svg)

> ${PUBLIC_PRODUCT_NAME} brand mark used inside the co-writer template.

### Lists

- ${PUBLIC_PRODUCT_NAME} Chat
- ${PUBLIC_PRODUCT_NAME} Co-Writer
- ${PUBLIC_PRODUCT_NAME} Research

1. Draft a concept note
2. Ask AI to refine it
3. Export the polished markdown

### Tables

Feature       | Description
------------- | -------------
Co-Writer     | Draft and refine Markdown content
Chat          | Ask questions and iterate ideas
Research      | Build structured multi-step reports

| Capability    | Primary Use Case                     |
| ------------- | ------------------------------------ |
| \`chat\`       | General tutoring and guidance        |
| \`deep_solve\` | Structured problem solving           |
| \`deep_question\` | Question generation and validation |

### Markdown extras

- [x] Draft a ${PUBLIC_PRODUCT_NAME} product note
- [x] Add references and structure
- [ ] Polish the final explanation
  - [ ] Check headings
  - [ ] Check citations

### TeX (LaTeX)

$$ E=mc^2 $$

Inline $$E=mc^2$$ appears in physics notes, and Inline $$a^2+b^2=c^2$$ appears in geometry notes.

$$\\sqrt{3x-1}+(1+x)^2$$

$$ \\sin(\\alpha)^{\\theta}=\\sum_{i=0}^{n}(x^i + \\cos(f))$$

### FlowChart

${FENCE}flow
st=>start: Student asks a question
op=>operation: ${PUBLIC_PRODUCT_NAME} analyzes intent
cond=>condition: Need deep workflow?
chat=>operation: Answer with chat capability
solve=>operation: Route to deep solve
e=>end: Return structured response

st->op->cond
cond(no)->chat
cond(yes)->solve
chat->e
solve->e
${FENCE}

### Sequence Diagram

${FENCE}seq
Student->${PUBLIC_PRODUCT_NAME}: Ask for help
${PUBLIC_PRODUCT_NAME}->KnowledgeBase: Load context
Note right of ${PUBLIC_PRODUCT_NAME}: Collect memory\\nand relevant knowledge
${PUBLIC_PRODUCT_NAME}-->Student: Return guided response
Student->>${PUBLIC_PRODUCT_NAME}: Request rewrite in co-writer
${FENCE}

### End
`;
