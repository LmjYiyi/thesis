---
name: thesis-figure-export
description: Standardize and export MATLAB figures for engineering thesis writing. Use this skill when asked to beautify plots, unify figure style across chapters, enforce axis labels/units, remove X-axis blank margins, apply publication-ready sizing, and export high-quality files (TIFF 300 dpi and EMF vector) directly from code.
---

# Thesis Figure Export

Apply a consistent, thesis-grade plotting style in MATLAB and export reproducible high-quality figures from code instead of manual UI operations.

## Workflow

1. Locate target plotting script and identify figure handles to standardize.
2. Add or reuse a local helper function `export_thesis_figure(...)`.
3. Enforce labels and units on both axes.
4. Set X-axis limits tightly to data range (no blank margins); allow small Y-axis padding.
5. Apply consistent typography, line widths, grid, and figure size:
   - Single-column: `14 cm x 8.65 cm` (approx.)
   - Double-column: `7 cm x 4.33 cm` (approx.)
6. Export both formats:
   - Raster: TIFF at 300 dpi
   - Vector: EMF
7. Save outputs to `figures_export/` under the current working directory.

## Implementation Rules

- Prefer Chinese labels for Chinese thesis figures.
- Mark normalized spectra as `归一化幅值 (无量纲)`; do not claim physical units unless calibrated.
- Use deterministic defaults:
  - `FontName`: `SimHei` (Chinese figures)
  - `FontSize`: `10`
  - axis line width: `1.0`
  - plot line width: `1.5`
  - `GridAlpha`: `0.20`
- Keep export logic inside script-local function for portability unless user requests a shared utility file.

## Resource

- Use `scripts/export_thesis_figure_template.m` as the base helper implementation.
