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
  --image-source-docx path/to/source-figures.docx ^
  --missing-images placeholder ^
  --output path/to/master_synced.docx
```

8. If the output path already exists, make sure Word is not keeping that file open. Otherwise write to a new filename first.
9. Open the output in Word and refresh the table of contents, figure index, and table index if needed.
10. Visually inspect formulas, tables, footnotes, hyperlinks, and chapter breaks.

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
  --level 2 ^
  --image-source-docx final_output/docs/thesis-master.docx ^
  --output path/to/master_synced.docx
```

- Overwrite in place with auto backup:

```bash
python skills/thesis-docx-sync/scripts/sync_markdown_to_docx.py ^
  --docx path/to/master.docx ^
  --markdown path/to/chapter.md ^
  --match-heading "Target Heading" ^
  --level 2 ^
  --in-place
```

## Resources

- Use `scripts/dump_docx_outline.py` to discover stable anchors before any edit.
- Use `scripts/build_markdown_chapter.py` when one logical chapter is stored as several Markdown files.
- Use `scripts/preflight_markdown_sync.py` to catch off-policy image paths and inline-math readability risks before rendering.
- Use `scripts/sync_markdown_to_docx.py` for section replacement.
- Use `scripts/docx_figure_catalog.py` to inspect which figure numbers a source DOCX can provide before relying on DOCX-backed image fallback.
- Read `references/workflow.md` when the user needs scope, constraints, or failure handling.

## Decision Notes

- If the requested fragment contains images or Word-only objects that are not ordinary Markdown images, split the task: sync text automatically, then place those objects manually in Word.
- If Markdown image file paths are broken because of historical encoding damage or missing asset folders, prefer `--image-source-docx` over hand-copying figures from Word.
- If the project already has a canonical figure directory, do not generate helper assets inside it just to satisfy DOCX insertion. Fix the Markdown path or stage conversions in a temp folder instead.
- If the chapter contains dense inline formulas, preflight them first and rewrite only the running-text expressions that hurt Word readability. Leave numbered display equations intact.
- If the requested change spans many chapters, still execute one heading-bounded section at a time to reduce corruption risk.
- If heading text differs between Markdown and Word, trust the existing Word heading and use it as the anchor.
- If the requested chapter exists only as several section Markdown files, use `scripts/build_markdown_chapter.py` to create a temporary merged chapter file before sync.
- If the output file is open in Word, write to a new output path or close Word before rerunning the sync.
- If the chapter updates headings, captions, bookmarks, or note references, refresh fields in Word before judging the table of contents or figure/table index pages.
