---
name: thesis-docx-sync
description: Update a thesis master .docx directly from revised Markdown chapters while preserving the document's existing heading and paragraph style system as much as possible. Use this skill when Codex needs to inspect a Word thesis outline, locate a chapter or section by heading text, and sync Markdown content back into the target .docx without manually copy-pasting into Word. Use only for .docx workflows; convert legacy .doc first.
---

# Thesis Docx Sync

Sync one heading-bounded section of a thesis `.docx` from Markdown instead of manually copying revised chapter text back into Word.

Keep the workflow narrow and safe: inspect the Word outline, target one existing heading, render Markdown with `pandoc --reference-doc`, and replace only that XML block inside the master document.

## Workflow

1. Confirm that the target file is `.docx`, not legacy `.doc`.
2. Dump the target outline:

```bash
python skills/thesis-docx-sync/scripts/dump_docx_outline.py path/to/master.docx
```

3. Copy the exact heading text from the outline output.
4. If one chapter is split across multiple Markdown files, merge them into a temporary chapter file first. Keep the temporary file near the source chapter files so relative image paths still resolve:

```bash
python skills/thesis-docx-sync/scripts/build_markdown_chapter.py ^
  --output writing/_chapter4_sync_tmp.md ^
  --title "Chapter 4 Title" ^
  writing/chapter4_4.1.md ^
  writing/chapter4_4.2.md ^
  writing/chapter4_4.3.md ^
  writing/chapter4_4.4.md
```

5. Run the Markdown preflight check before sync. For this thesis project, the recommended profile is `final_output_doc`, which enforces local images from `figures/final_output_doc/` and flags risky inline math:

```bash
python skills/thesis-docx-sync/scripts/preflight_markdown_sync.py ^
  --markdown writing/_chapter4_sync_tmp.md ^
  --profile final_output_doc ^
  --fail-on-warning
```

6. When the master `.docx` is an old template, verify the target block by position and neighboring chapter headings, not by chapter number alone. Do not assume the requested Markdown chapter maps to the current fourth `Heading 1` in Word.
7. Sync the Markdown section back into the master document:

```bash
python skills/thesis-docx-sync/scripts/sync_markdown_to_docx.py ^
  --docx path/to/master.docx ^
  --markdown path/to/chapter.md ^
  --match-heading "Target Heading" ^
  --level 1 ^
  --missing-images placeholder ^
  --output path/to/master_synced.docx
```

8. **After syncing all chapters**, run the mandatory comprehensive post-processor. This step is **required** — without it, body text uses wrong styles, tables lack formatting, captions are unstyled, and images may be clipped:

```bash
python skills/thesis-docx-sync/scripts/postprocess_styles.py ^
  --docx path/to/master_synced.docx ^
  --in-place
```

9. If the output path already exists, make sure Word is not keeping that file open. Otherwise write to a new filename first.
10. Open the output in Word and refresh the table of contents, figure index, and table index if needed.
11. Visually inspect formulas, tables, footnotes, hyperlinks, and chapter breaks.
12. Clean up temporary files:
    - Delete any temporary outline dump files (e.g., `temp_outline.txt`) created during the workflow
    - Delete auto-generated backup files created by `postprocess_styles.py` (e.g., `*.bak-*` files)

## Rules

- Treat the target `.docx` as the layout authority.
- Use `--reference-doc` against the target `.docx` or a style-identical template so `pandoc` emits compatible heading and paragraph styles.
- Replace only one heading-bounded section at a time.
- Prefer chapter or section headings that already exist in the master Word file.
- When syncing into a legacy thesis template, identify the replacement block from the actual Word outline, not from the intended chapter number in Markdown.
- If one logical chapter is stored as multiple Markdown files, merge them before sync and shift all heading levels down by one under a chapter-level heading.
- Keep merged temporary Markdown next to the original chapter files when the source uses relative image paths such as `figures/...`.
- Run `preflight_markdown_sync.py` before every real sync when the chapter contains figures or dense math.
- For this thesis project, treat `writing/figures/final_output_doc/` as the canonical figure library. Insert only those files unless the user explicitly changes that rule.
- Do not create helper images such as `sync_fig*.png` or `_preview_*.png` inside the canonical figure directory. Temporary conversions, previews, and staged copies must live under a temp directory only.
- Prefer explicit local image paths such as `figures/final_output_doc/fig3.7.tiff` when the final figure library already exists.
- In running text, flatten simple inline fractions to slash notation such as `$d\tau_g/df$`, `$2\pi B/T_m$`, or `$d(f_p/f)^2/(2c)$` when that improves Word readability.
- Keep numbered derivations and display equations as block equations. Do not flatten numbered formulas solely for convenience.
- Avoid display-style constructs such as `\frac`, `\dfrac`, `\underbrace`, `\left...\right`, or long nested expressions inside inline math when a simpler body-text form is available.
- Strip Markdown heading prefixes such as `第二章`, `2.1`, or `2.4.1` before rendering when the target Word template already applies automatic heading numbering.
- Allow front-matter anchors such as `摘要` or `ABSTRACT` when the Word template uses a non-numbered title style instead of `Heading 1`.
- Allow local Markdown image references and display equations with `\tag{...}`; the script preprocesses those two cases before rendering.
- If a Markdown image path is broken or the user wants to reuse figures that only exist inside another Word thesis, pass `--image-source-docx`. The script will fall back to that DOCX and match figures by number such as `图5-1`.
- Allow explicit DOCX-backed image placeholders like `![图5-1 宽带LFMCW诊断系统扩频后的射频前端链路架构](docx-figure://图5-1)` when the source Markdown should not depend on a local image file path.
- If source image files are missing but the text sync still needs to proceed, use `--missing-images placeholder` to replace those image nodes with visible placeholder text in the output DOCX.
- Allow footnotes, endnotes, external hyperlinks, and internal `#anchor` jump links generated by `pandoc`.
- Align the first inserted heading paragraph to the target heading style so front-matter titles keep the template's existing formatting.
- Treat the table of contents and figure/table indexes as field-generated pages. Update them in Word after sync instead of editing those pages manually.
- Keep tracked changes, comments, and embedded OLE objects out of the Markdown fragment.
- Do not assume equation numbers will land in exactly the same visual position as a hand-crafted Word equation template; verify the layout after sync.
- If the user asks to modify a legacy `.doc`, convert it to `.docx` first instead of attempting direct `.doc` editing.
- **IMPORTANT: `postprocess_styles.py` is mandatory after every sync.** It now handles ALL of the following:
  - Body text style remapping (pandoc → Normal)
  - Figure caption detection (after image paragraphs → `标题-图`)
  - Table caption detection (before `w:tbl` elements → `标题-表格`)
  - Reference section styling (after `参考文献` heading → `参考文献`)
  - Display equation styling (paragraphs with `m:oMathPara` → `公式`)
  - Table formatting: 三线表 style (thick top/bottom borders, thin header separator, no vertical lines, centered, 10.5pt font)
  - Image paragraph formatting: centered, auto line spacing
  - Page header updates: correct "第X章 标题" text per chapter
- When syncing multiple chapters in sequence, sync ALL chapters first, then run `postprocess_styles.py` **once** on the final output. Running it per-chapter is also fine but unnecessary.
- The `--level` CLI argument for `sync_markdown_to_docx.py` only accepts values 1–9. For level-0 headings (e.g., `摘要`, `ABSTRACT` with `标题-无编号` style), **omit** the `--level` flag entirely — the heading text alone is sufficient for unique matching.

## Markdown Preparation Rules

**CRITICAL: Figure and table captions must NOT include number prefixes.**

The Word template's `标题-图` and `标题-表格` styles apply automatic numbering (图3.1, 表4.2, etc.). If the Markdown caption also includes the number, the output will show **duplicated numbering** like "图3.1 图3.1 caption text".

- Figure captions: Write `![caption text](path)` — not `![图3.1 caption text](path)`
- Table captions: Write just `caption text` on the line before the table — not `表4.1 caption text`
- Bold-wrapped captions like `**表4.2 title**` must also be stripped to just `title`

**参考文献.md must have a heading.**

The file `writing/参考文献.md` must start with `# 参考文献` on the first line. Without it, the sync script replaces the heading text in Word with the first reference entry.

**Complex inline math may be invisible in Word.**

Inline math (`$...$`) containing `\frac`, `\sqrt`, or `\boxed` may render as invisible zero-height OMML in Word. Before sync:
- Convert complex inline fractions to display math blocks (`$$...$$`)
- Or simplify to slash notation: `$\frac{a}{b}$` → `$a/b$`
- Remove `\boxed{}` from any formula (use plain display math instead)

## Commands

- Dump outline:

```bash
python skills/thesis-docx-sync/scripts/dump_docx_outline.py path/to/master.docx
```

- Merge split Markdown files into one chapter file:

```bash
python skills/thesis-docx-sync/scripts/build_markdown_chapter.py ^
  --output path/to/chapter_merged.md ^
  --title "Chapter Title" ^
  path/to/section1.md ^
  path/to/section2.md
```

- Preflight Markdown before sync:

```bash
python skills/thesis-docx-sync/scripts/preflight_markdown_sync.py ^
  --markdown path/to/chapter_merged.md ^
  --profile final_output_doc ^
  --fail-on-warning
```

- Sync to a new output file:

```bash
python skills/thesis-docx-sync/scripts/sync_markdown_to_docx.py ^
  --docx path/to/master.docx ^
  --markdown path/to/chapter.md ^
  --match-heading "Target Heading" ^
  --level 1 ^
  --missing-images placeholder ^
  --output path/to/master_synced.docx
```

- Post-process (mandatory, comprehensive):

```bash
python skills/thesis-docx-sync/scripts/postprocess_styles.py ^
  --docx path/to/synced.docx ^
  --in-place
```

- Post-process without header updates:

```bash
python skills/thesis-docx-sync/scripts/postprocess_styles.py ^
  --docx path/to/synced.docx ^
  --no-headers ^
  --in-place
```

## Resources

- Use `scripts/dump_docx_outline.py` to discover stable anchors before any edit.
- Use `scripts/build_markdown_chapter.py` when one logical chapter is stored as several Markdown files.
- Use `scripts/preflight_markdown_sync.py` to catch off-policy image paths and inline-math readability risks before rendering.
- Use `scripts/sync_markdown_to_docx.py` for section replacement.
- Use `scripts/docx_figure_catalog.py` to inspect which figure numbers a source DOCX can provide before relying on DOCX-backed image fallback.
- Use `scripts/postprocess_styles.py` **after every sync** — it is the single comprehensive post-processor that handles style remapping, caption detection, table formatting (三线表), image layout, and page headers.
- Read `references/workflow.md` when the user needs scope, constraints, or failure handling.

## Decision Notes

**IMPORTANT: Abstract files must be split into separate files.**
- The Chinese abstract (`摘要_中文.md`) and English abstract (`摘要_英文.md`) MUST be stored as separate Markdown files.
- Each abstract file must contain ONLY ONE top-level heading (`# 摘要` or `# ABSTRACT`).
- NEVER combine both abstracts in a single file — this will cause heading conflicts during sync and result in duplicate "Abstract" entries in the Word document.
- When syncing abstracts, **omit** the `--level` flag (the CLI does not accept `--level 0`). The heading text `摘要` or `ABSTRACT` is unique enough for matching.
- Sync order: Chinese abstract first, then English abstract.

**IMPORTANT: Chapter title matching.**
- When merging multiple Markdown files into one chapter, use the FULL chapter title (e.g., "第三章 宽带信号在色散介质中的传播机理与方法失效分析") as the `--title` parameter, NOT just "第3章".
- The merged file's first heading must match the Word template's existing chapter heading text.

**IMPORTANT: Full multi-chapter sync workflow.**

When syncing all chapters at once, follow this order:
1. Copy the master `.docx` to a working copy (never overwrite the master directly).
2. Sync abstracts: Chinese (`--match-heading "摘要"`, no `--level`) → English (`--match-heading "ABSTRACT"`, no `--level`).
3. Sync chapters 1–6 sequentially, each with `--in-place` on the working copy. Use `--match-heading` with the **exact heading text from the Word outline**, not the Markdown heading.
4. Sync `参考文献` (`--match-heading "参考文献"`, no `--level`).
5. Run `postprocess_styles.py --in-place` once on the final working copy.
6. Open in Word, refresh TOC / figure index / table index, inspect images and tables.

**IMPORTANT: Heading-to-file mapping for this thesis.**

| Word heading text (--match-heading) | --level | Markdown source |
|---|---|---|
| 摘要 | (omit) | writing/摘要_中文.md |
| ABSTRACT | (omit) | writing/摘要_英文.md |
| 绪论 | 1 | writing/第1章_绪论_v1.md |
| 等离子体电磁特性与LFMCW诊断机理 | 1 | writing/第2章_...final.md |
| 宽带信号在色散介质中的传播机理与误差量化 | 1 | (merge 第3章_3.2~3.5_final.md) |
| 系统差频信号数据处理 | 1 | (merge 第4章_4.2~4.5_final.md) |
| 系统标定及诊断实验 | 1 | (merge 第5章_5.2~5.5_final.md) |
| 总结与展望 | 1 | writing/第6章_总结与展望_final.md |
| 参考文献 | (omit) | writing/参考文献.md |

**IMPORTANT: Pandoc style mismatch — postprocess_styles.py is mandatory.**

Full style remapping table:

| pandoc generates | postprocess remaps to | Detection method |
|---|---|---|
| `Body Text` (a8) | Normal（正文） | Style ID match |
| `FirstParagraph` | Normal（正文） | Style ID match |
| `Compact` | Normal（正文） | Style ID match |
| `CaptionedFigure` | Normal（正文） | Style ID match |
| `Figure` | Normal（正文） | Style ID match |
| `ImageCaption` | `标题-图` (-0) | Style ID match |
| (paragraph after image) | `标题-图` (-0) | Position: prev sibling has drawing |
| (paragraph before table) | `标题-表格` (-) | Position: next sibling is w:tbl |
| (after 参考文献 heading) | `参考文献` (a) | Section: after -1 styled 参考文献 |
| (contains m:oMathPara) | `公式` (aff1) | Content: has display math |
| (all w:tbl elements) | 三线表 formatting | Element type |

- If the requested fragment contains images or Word-only objects that are not ordinary Markdown images, split the task: sync text automatically, then place those objects manually in Word.
- If Markdown image file paths are broken because of historical encoding damage or missing asset folders, prefer `--image-source-docx` over hand-copying figures from Word.
- If the project already has a canonical figure directory, do not generate helper assets inside it just to satisfy DOCX insertion. Fix the Markdown path or stage conversions in a temp folder instead.
- If the chapter contains dense inline formulas, preflight them first and rewrite only the running-text expressions that hurt Word readability. Leave numbered display equations intact.
- If the requested change spans many chapters, still execute one heading-bounded section at a time to reduce corruption risk.
- If heading text differs between Markdown and Word, trust the existing Word heading and use it as the anchor.
- If the requested chapter exists only as several section Markdown files, use `scripts/build_markdown_chapter.py` to create a temporary merged chapter file before sync.
- If the output file is open in Word, write to a new output path or close Word before rerunning the sync.
- If the chapter updates headings, captions, bookmarks, or note references, refresh fields in Word before judging the table of contents or figure/table index pages.
