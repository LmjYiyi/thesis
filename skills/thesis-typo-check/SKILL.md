---
name: thesis-typo-check
description: Check thesis text for typos, missing or repeated characters, obvious wording mistakes, punctuation issues, and local terminology inconsistencies. Use this skill when Codex is asked to proofread Chinese or English thesis material in Markdown, copied DOCX text, plain text, abstracts, chapter drafts, or review replies with a typo-focused scope rather than a full rewrite.
---

# Thesis Typo Check

Review thesis prose with a narrow proofreading scope: catch wrong characters, missing characters, duplicate words, punctuation slips, and clearly incorrect wording without rewriting the author's style.

Prefer precision over coverage. Report only issues that are certain or strongly justified by nearby context.

## Workflow

1. Confirm the scope from the request:
   - `typo-only`: focus on wrong characters, missing characters, repeated characters, punctuation, spacing, and obvious wording slips.
   - `light-proofread`: include short local grammar fixes when the sentence is clearly broken.
2. Read the text in manageable chunks. For long files, work section by section instead of scanning the whole thesis as one block.
3. Preserve structure while reading:
   - Ignore Markdown headings, list markers, and table syntax unless they contain visible typos.
   - Ignore formulas, citation keys, file paths, URLs, code, and raw LaTeX unless the user explicitly asks to proofread them too.
4. Identify only concrete issues you can point to exactly.
5. Return findings as a compact list or table with:
   - location
   - original text
   - suggested fix
   - short reason
   - confidence when the fix is uncertain
6. Provide a fully corrected passage only if the user asks for direct rewriting after the issue list.

## Rules

- Keep the task narrow. Do not silently upgrade typo checking into stylistic rewriting.
- Quote the exact problematic fragment so the user can find it quickly.
- Distinguish `certain` from `suspected` issues.
- Prefer sentence-local evidence. Only use broader terminology consistency checks when the same section already establishes the intended term.
- Preserve domain terminology unless there is a clear typo:
  - `LFMCW`, `Drude`, `Lorentz`, `Metropolis-Hastings`, `ESPRIT`, `MATLAB`, `Markdown`, `Word`
- Preserve symbols, variable names, units, and equation references unless the surrounding prose makes the typo obvious.
- For Chinese text, check missing particles, duplicated characters, wrong homophones, wrong near-shape characters, and broken punctuation pairing.
- For mixed Chinese-English technical writing, check spacing and case consistency around English terms, abbreviations, and units.
- When the text is ambiguous, say so instead of guessing.

## Output Format

Use this compact format by default:

```text
1. [Location]
   Original: ...
   Suggestion: ...
   Reason: ...
   Confidence: high/medium
```

If there are many issues in one section, use a table with columns `Location | Original | Suggestion | Reason | Confidence`.

If no clear issues are found, say that explicitly and mention any residual risk such as unchecked formulas, references, or terminology outside the provided excerpt.

## Typical Requests

- `Use $thesis-typo-check to check this chapter for typos and only mark issues.`
- `Check this abstract for wrong or missing characters and obvious broken sentences.`
- `Proofread this Markdown chapter with attention to terminology spelling and punctuation.`

## Resources

- Read `references/checklist.md` when you need a reminder of high-frequency thesis proofreading checks, terminology consistency traps, or a stricter review checklist.
