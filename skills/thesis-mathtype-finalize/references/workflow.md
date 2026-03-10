# MathType Finalization Workflow

## Scope

This skill handles one narrow job: take a thesis `.docx` that already contains the final synced content and convert Word OMML equations into MathType objects through the locally installed Word + MathType integration.

It is intended for:

- inline equations inside body text
- display equations
- numbered display equations that were previously inserted as Word equations

It is not intended for:

- legacy `.doc`
- daily chapter sync
- tracked changes cleanup
- comments or OLE repair
- fully headless automation without a desktop Word session

## Why This Is Separate From DOCX Sync

`thesis-docx-sync` solves content replacement. It keeps Markdown, images, headings, links, and OMML equations aligned with the thesis template.

MathType conversion is a different stage:

- it depends on local Office automation
- it can change pagination and line spacing
- it is usually only required near submission

Keep the two stages separate so ordinary writing stays stable.

## Environment Assumptions

- Windows machine
- Microsoft Word installed
- MathType installed
- MathType Word template available in the Word startup path

During development, the installed Office template exposed these macro entry points:

- `MathTypeCommands.UILib.MTCommand_ConvertEqns`
- `MathTypeCommands.MTConvertEquations.DlgMain`
- `MathTypeCommands.MTCommandsDispatchCls.MTCommandsMain_MTConvertEquations_DlgMain`

The probe script checks for those names by reading the installed `MathType Commands 2016.dotm` template.

## Recommended Flow

1. Finish Markdown edits and sync them into a versioned `.docx`.
2. Probe the environment:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\probe_mathtype_env.ps1
```

3. Run the finalize script against a copied version:

```powershell
powershell -ExecutionPolicy Bypass -File skills\thesis-mathtype-finalize\scripts\convert_docx_to_mathtype.ps1 `
  -InputDocx writing\versions\v4.docx `
  -OutputDocx writing\versions\v5.docx
```

4. When Word shows the MathType conversion dialog, choose:
   - whole document
   - OMML equations
   - MathType equations
5. Let the conversion finish in Word.
6. Save the document.
7. Inspect representative equations across:
   - abstract or front matter
   - inline formulas inside paragraphs
   - long display equations
   - numbered equations near page breaks

## Verification

The conversion script reports the OMML count before and after save by reading `word/document.xml` inside the `.docx`.

Interpretation:

- before > 0 and after = 0: good signal that OMML equations were converted
- before > 0 and after is still large: conversion likely did not run or did not target OMML
- before = 0: the document may already contain MathType or other non-OMML equation objects

Still do a visual check in Word. XML counts alone do not guarantee perfect layout.

## Failure Modes

- Word COM cannot start: run the conversion from a normal desktop user session instead of a restricted terminal session.
- MathType template not found: repair the MathType Word plugin installation.
- The conversion dialog opens but no equations change: confirm the dialog options were `OMML -> MathType` and the scope was the whole document.
- The file remains locked after conversion: close the document in Word or rerun with a new output filename.
- Page breaks shift: accept that as a finalization-side effect and do a manual layout pass afterward.

## Practical Advice

- Use this only on versioned output files such as `v5.docx`, `v6.docx`.
- Do not overwrite the previous good version.
- Convert after major content changes settle down.
- If the school does not strictly require MathType objects, keep the OMML version as the primary editable source of truth.
