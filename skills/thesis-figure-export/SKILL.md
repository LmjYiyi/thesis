---
name: thesis-figure-export
description: Standardize and export MATLAB figures for engineering thesis writing. Use this skill when asked to beautify plots, unify figure style across chapters, enforce axis labels/units, remove layout conflicts, and export high-quality files (TIFF 300 dpi and EMF vector) directly from code.
---

# Thesis Figure Export

Apply a consistent, thesis-grade plotting style in MATLAB and export reproducible high-quality figures from code instead of manual UI operations.

## Workflow

1. Locate target plotting script and identify figure handles to standardize.
2. Add or reuse a local helper function `export_thesis_figure(...)`.
3. Remove in-figure global title (`sgtitle`/`suptitle`) from export. Put overall title in thesis caption, not inside image.
4. Enforce labels and units on both axes.
5. Handle legend without covering key curves:
   - Prefer in-axis placement (`best`) for single-axis figures.
   - Use outside legend only when necessary for readability.
6. Apply consistent typography, line widths, grid, and figure size:
   - Single-column recommended: `14 cm x 9.1 cm` (for 1x2 subplot figures)
   - Double-column recommended: `7 cm x 4.55 cm`
7. Export both formats:
   - Raster: TIFF at 600 dpi (preferred), 300 dpi minimum
   - Vector: EMF
8. Save outputs to `figures_export/` under the current working directory.

## Implementation Rules

- Prefer Chinese labels for Chinese thesis figures.
- Do not claim physical units for normalized spectra unless calibrated.
- Use deterministic defaults:
  - Chinese text font: `SimSun`
  - Tick/number font: `Times New Roman`
  - `FontSize`: `10`
  - axis line width: `1.0`
  - plot line width: `1.5`
  - `GridAlpha`: `0.25`
  - `TickDir`: `in`
- Layout quality constraints (mandatory):
  - Do not force `tighten_axis_limits` on `yyaxis` figures.
  - Keep enough top and bottom margin to avoid title/legend overlap.
- Legend quality constraints (mandatory):
  - `Box`: `off`
  - `AutoUpdate`: `off`
  - do not force `southoutside` as default
- Keep export logic inside script-local function for portability unless user requests a shared utility file.

## Resource

- Use `scripts/export_thesis_figure_template.m` as the base helper implementation.
