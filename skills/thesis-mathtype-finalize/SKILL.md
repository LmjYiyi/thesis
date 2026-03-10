---
name: thesis-mathtype-finalize
description: Convert synced thesis .docx equations from Word OMML into MathType objects by driving the local Windows Microsoft Word + MathType integration. Use when a chapter or full thesis has already been synced into a .docx and the user needs both inline equations and display equations converted to MathType as a finalization pass.
---

# Thesis MathType Finalize

Finalize a synced thesis `.docx` by converting Word equations into MathType objects.

Use this skill only after Markdown-to-DOCX sync is already finished. Treat it as a final formatting pass, not as part of the daily writing loop.

## Workflow

1. Probe the local environment first:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\probe_mathtype_env.ps1
```

2. Start from a copied `.docx` version, usually under `writing\versions\`.
3. Run the finalize script:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\convert_docx_to_mathtype.ps1 `
  -InputDocx writing\versions\v4.docx `
  -OutputDocx writing\versions\v5.docx
```

4. Let Word open the MathType conversion dialog.
5. In the dialog, choose:
   - scope: whole document
   - from: OMML equations
   - to: MathType equations
6. Finish the conversion in Word, then let the script save and close the output file.
7. Visually inspect several inline equations, display equations, and numbered equations.

## Rules

- Work only on `.docx`, not `.doc`.
- Write to a new versioned file. Do not run this against the only master copy.
- Keep the target document free of tracked changes before conversion.
- Prefer the built-in MathType conversion dialog over fragile UI emulation.
- Expect possible layout drift in line spacing, equation number placement, and page breaks after conversion.
- If Word COM or the MathType template is unavailable, stop and fall back to manual conversion inside Word.

## Commands

- Probe local Word/MathType availability:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\probe_mathtype_env.ps1
```

- Convert a synced DOCX into a new finalized version:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\convert_docx_to_mathtype.ps1 `
  -InputDocx writing\versions\v4.docx `
  -OutputDocx writing\versions\v5.docx
```

- Keep Word open after conversion for manual inspection:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\convert_docx_to_mathtype.ps1 `
  -InputDocx writing\versions\v4.docx `
  -OutputDocx writing\versions\v5.docx `
  -KeepWordOpen
```

## Resources

- Use `scripts/probe_mathtype_env.ps1` to confirm Word COM, MathType template presence, and callable macro entry points.
- Use `scripts/convert_docx_to_mathtype.ps1` to open the target document, invoke the MathType conversion command, and save a new version.
- Read `references/workflow.md` when you need prerequisites, expected dialog choices, or failure handling details.

## Decision Notes

- If the user only wants equations to remain editable in Word, prefer the existing OMML workflow and skip MathType conversion.
- If the user needs school-submission formatting or advisor-required MathType objects, run this skill as the final pass on the synced `.docx`.
- If the thesis has only one changed chapter, still convert the full copied `.docx` output so equation numbering and layout can be checked in context.
