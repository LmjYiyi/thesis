"""Post-process a synced thesis DOCX to remap pandoc-generated styles
back to the reference document's style set.

Style remapping rules (based on reference doc analysis):
  - a8  (Body Text)       → remove pStyle (= Normal/正文)
  - FirstParagraph        → remove pStyle (= Normal/正文)
  - Compact               → remove pStyle (= Normal/正文)
  - ImageCaption          → -0 (标题-图)
  - CaptionedFigure       → remove pStyle (= Normal/正文, image container)
  - Figure                → remove pStyle
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from collections import Counter

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W_NS, "wp": WP_NS}

# Pandoc styles that should become Normal (remove pStyle element)
REMAP_TO_NORMAL = {"a8", "FirstParagraph", "Compact", "CaptionedFigure", "Figure"}
# Pandoc styles that should become 标题-图
REMAP_TO_FIGURE_CAPTION = {"ImageCaption"}


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    nsmap = {"w": W_NS, "wp": WP_NS}
    return f"{{{nsmap[prefix]}}}{local}"


def _ensure_ppr(p):
    """Get or create w:pPr as the first child of a w:p element."""
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.SubElement(p, qn("w:pPr"))
        # Move pPr to be first child
        p.remove(ppr)
        p.insert(0, ppr)
    return ppr


def _has_drawing(p):
    """Check if paragraph contains any inline image (w:drawing with wp:inline)."""
    return (
        len(p.findall(f".//{{{WP_NS}}}inline")) > 0
        or len(p.findall(f".//{{{WP_NS}}}anchor")) > 0
    )


def _fix_image_paragraph_format(ppr, stats):
    """Apply reference-doc image paragraph formatting:
    - spacing: line=240, lineRule=auto (single spacing, auto height)
    - jc: center
    This ensures the image is not clipped by fixed line height.
    """
    # Set or replace spacing
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.SubElement(ppr, qn("w:spacing"))
    spacing.set(qn("w:line"), "240")
    spacing.set(qn("w:lineRule"), "auto")

    # Set or replace jc (center alignment)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, qn("w:jc"))
    jc.set(qn("w:val"), "center")

    stats["image para → auto spacing + center"] += 1


def process(docx_path: Path, output_path: Path) -> dict:
    stats: Counter = Counter()

    with zipfile.ZipFile(docx_path, "r") as zin:
        doc_xml = zin.read("word/document.xml")

    root = ET.fromstring(doc_xml)
    body = root.find(".//w:body", NS)
    if body is None:
        raise RuntimeError("No w:body found")

    for p in body.iter(qn("w:p")):
        ppr = p.find("w:pPr", NS)

        # --- Style remapping ---
        if ppr is not None:
            ps = ppr.find("w:pStyle", NS)
            if ps is not None:
                sid = ps.get(qn("w:val"))
                if sid and sid in REMAP_TO_NORMAL:
                    ppr.remove(ps)
                    stats[f"{sid} → Normal"] += 1
                elif sid and sid in REMAP_TO_FIGURE_CAPTION:
                    ps.set(qn("w:val"), "-0")
                    stats[f"{sid} → 标题-图"] += 1

        # --- Fix image paragraphs: ensure auto line spacing + center ---
        if _has_drawing(p):
            ppr = _ensure_ppr(p)
            _fix_image_paragraph_format(ppr, stats)

    # Re-serialize
    # Register all namespaces to avoid ns0/ns1 prefix pollution
    namespaces_to_register = [
        ("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
        ("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
        ("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
        ("a", "http://schemas.openxmlformats.org/drawingml/2006/main"),
        ("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture"),
        ("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006"),
        ("o", "urn:schemas-microsoft-com:office:office"),
        ("v", "urn:schemas-microsoft-com:vml"),
        ("m", "http://schemas.openxmlformats.org/officeDocument/2006/math"),
        ("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"),
        ("w14", "http://schemas.microsoft.com/office/word/2010/wordml"),
        ("w15", "http://schemas.microsoft.com/office/word/2012/wordml"),
        ("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"),
        ("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"),
        ("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"),
        ("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk"),
    ]
    for prefix, uri in namespaces_to_register:
        ET.register_namespace(prefix, uri)

    new_doc_xml = ET.tostring(root, encoding="unicode", xml_declaration=True)

    # Write output DOCX by copying everything except document.xml
    with zipfile.ZipFile(docx_path, "r") as zin, \
         zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, new_doc_xml.encode("utf-8"))
            else:
                zout.writestr(item, zin.read(item.filename))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Remap pandoc styles to reference doc styles")
    parser.add_argument("--docx", required=True, type=Path, help="Input DOCX")
    parser.add_argument("--output", type=Path, help="Output DOCX (default: overwrite with backup)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input with backup")
    args = parser.parse_args()

    if args.in_place:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.docx.with_name(f"{args.docx.stem}.bak-{ts}{args.docx.suffix}")
        shutil.copy2(args.docx, backup)
        output = args.docx
    elif args.output:
        output = args.output
    else:
        output = args.docx.with_name(f"{args.docx.stem}_styled{args.docx.suffix}")

    # If in-place, work on a temp copy
    if args.in_place:
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".docx"))
        shutil.copy2(args.docx, tmp)
        stats = process(tmp, output)
        tmp.unlink()
    else:
        stats = process(args.docx, output)

    print(f"output: {output}")
    print(f"style remapping stats:")
    for k, v in stats.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
