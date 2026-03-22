"""Post-process a synced thesis DOCX: style remapping, table formatting,
caption detection, page headers, and image paragraph layout.

This is the **mandatory** post-processor after every sync.  It supersedes the
earlier minimal version and incorporates all lessons from multi-session sync
work (March 2024–2026).

Style remapping (pandoc → thesis template):
  Body Text / FirstParagraph / Compact / BodyText → Normal（正文）
  ImageCaption → 标题-图 (-0)
  CaptionedFigure / Figure → Normal

Position-based caption detection:
  Paragraph immediately after an image paragraph → 标题-图 (-0)
  Paragraph immediately before a w:tbl element → 标题-表格 (-)

Section-based detection:
  All paragraphs after 参考文献 heading → 参考文献 (a)

Display math:
  Paragraphs containing m:oMathPara → 公式 (aff1)

Table formatting:
  三线表 style: top/bottom thick borders, header bottom thin border,
  no vertical lines, centered alignment, 10.5pt font.

Page headers:
  Detect Heading 1 paragraphs, update corresponding header XML files
  with correct "第X章 标题" text.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── Namespaces ──────────────────────────────────────────────────────────
NAMESPACES = {
    "w":    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r":    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp":   "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a":    "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic":  "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "v":    "urn:schemas-microsoft-com:vml",
    "o":    "urn:schemas-microsoft-com:office:office",
    "mc":   "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "m":    "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "wps":  "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "ct":   "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel":  "http://schemas.openxmlformats.org/package/2006/relationships",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "w14":  "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15":  "http://schemas.microsoft.com/office/word/2012/wordml",
    "wpc":  "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    "wpg":  "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "wpi":  "http://schemas.microsoft.com/office/word/2010/wordprocessingInk",
}

for _pfx, _uri in NAMESPACES.items():
    ET.register_namespace(_pfx, _uri)

W_NS = NAMESPACES["w"]
R_NS = NAMESPACES["r"]
WP_NS = NAMESPACES["wp"]
M_NS = NAMESPACES["m"]

# Pandoc styles → Normal (remove pStyle)
REMAP_TO_NORMAL = {"a8", "FirstParagraph", "Compact", "CaptionedFigure", "Figure", "BodyText"}
# Pandoc styles → 标题-图
REMAP_TO_FIGURE_CAPTION = {"ImageCaption"}

# Chapter header titles (keyed by order of appearance)
CHAPTER_HEADERS = [
    "\u7b2c\u4e00\u7ae0 \u7eea\u8bba",                                             # 第一章 绪论
    "\u7b2c\u4e8c\u7ae0 \u7b49\u79bb\u5b50\u4f53\u7535\u78c1\u7279\u6027\u4e0eLFMCW\u8bca\u65ad\u673a\u7406",  # 第二章 ...
    "\u7b2c\u4e09\u7ae0 \u5bbd\u5e26\u4fe1\u53f7\u5728\u8272\u6563\u4ecb\u8d28\u4e2d\u7684\u4f20\u64ad\u673a\u7406\u4e0e\u8bef\u5dee\u91cf\u5316",
    "\u7b2c\u56db\u7ae0 \u7cfb\u7edf\u5dee\u9891\u4fe1\u53f7\u6570\u636e\u5904\u7406",
    "\u7b2c\u4e94\u7ae0 \u7cfb\u7edf\u6807\u5b9a\u53ca\u8bca\u65ad\u5b9e\u9a8c",
    "\u7b2c\u516d\u7ae0 \u603b\u7ed3\u4e0e\u5c55\u671b",
]


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    return "{%s}%s" % (NAMESPACES[prefix], local)


def get_text(elem) -> str:
    return "".join(t.text for t in elem.iter(qn("w:t")) if t.text)


def get_style(p) -> str | None:
    pPr = p.find(qn("w:pPr"))
    if pPr is not None:
        ps = pPr.find(qn("w:pStyle"))
        if ps is not None:
            return ps.get(qn("w:val"))
    return None


def has_drawing(p) -> bool:
    return (p.find(".//" + qn("wp:inline")) is not None
            or p.find(".//" + qn("wp:anchor")) is not None)


def set_pstyle(p, style_id: str):
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        pPr = ET.SubElement(p, qn("w:pPr"))
        p.insert(0, pPr)
    ps = pPr.find(qn("w:pStyle"))
    if ps is None:
        ps = ET.SubElement(pPr, qn("w:pStyle"))
        pPr.insert(0, ps)
    ps.set(qn("w:val"), style_id)


def remove_pstyle(p):
    pPr = p.find(qn("w:pPr"))
    if pPr is not None:
        ps = pPr.find(qn("w:pStyle"))
        if ps is not None:
            pPr.remove(ps)


# ═══════════════════════════════════════════════════════════════════════
# 1. STYLE REMAPPING + CAPTION DETECTION
# ═══════════════════════════════════════════════════════════════════════

def remap_styles(body) -> Counter:
    """Remap pandoc styles and detect captions by position."""
    stats: Counter = Counter()
    children = list(body)

    for i, elem in enumerate(children):
        if elem.tag != qn("w:p"):
            continue

        style = get_style(elem)
        text = get_text(elem).strip()

        # ── Pandoc style remapping ──
        if style in REMAP_TO_NORMAL:
            remove_pstyle(elem)
            stats["normal_reset"] += 1

        if style in REMAP_TO_FIGURE_CAPTION:
            set_pstyle(elem, "-0")
            stats["fig_caption"] += 1
        elif not has_drawing(elem) and text and not text.startswith("|"):
            # Paragraph right after an image → figure caption
            if i > 0 and children[i - 1].tag == qn("w:p") and has_drawing(children[i - 1]):
                set_pstyle(elem, "-0")
                stats["fig_caption"] += 1

        # ── Table caption: paragraph right before a w:tbl ──
        if i + 1 < len(children):
            nxt = children[i + 1]
            if nxt.tag == qn("w:tbl") and text and style != "1":
                set_pstyle(elem, "-")
                stats["tbl_caption"] += 1

        # ── Display math: m:oMathPara ──
        if elem.find(".//" + qn("m:oMathPara")) is not None:
            set_pstyle(elem, "aff1")
            stats["equation"] += 1

    # ── Reference section ──
    in_ref = False
    for elem in children:
        if elem.tag != qn("w:p"):
            continue
        style = get_style(elem)
        text = get_text(elem).strip()
        if style == "-1" and "\u53c2\u8003\u6587\u732e" in text:
            in_ref = True
            continue
        if in_ref:
            if style in ("1", "-1"):
                in_ref = False
                continue
            if text:
                set_pstyle(elem, "a")
                stats["reference"] += 1

    # ── Cleanup: remove false-positive table captions ──
    for i, elem in enumerate(children):
        if elem.tag == qn("w:p") and get_style(elem) == "-":
            if i + 1 < len(children) and children[i + 1].tag != qn("w:tbl"):
                remove_pstyle(elem)
                stats["tbl_caption"] -= 1

    return stats


# ═══════════════════════════════════════════════════════════════════════
# 2. IMAGE PARAGRAPH FORMATTING
# ═══════════════════════════════════════════════════════════════════════

def format_image_paragraphs(body) -> int:
    """Center image paragraphs with auto line spacing."""
    count = 0
    for p in body.iter(qn("w:p")):
        if not has_drawing(p):
            continue
        pPr = p.find(qn("w:pPr"))
        if pPr is None:
            pPr = ET.SubElement(p, qn("w:pPr"))
            p.insert(0, pPr)

        jc = pPr.find(qn("w:jc"))
        if jc is None:
            jc = ET.SubElement(pPr, qn("w:jc"))
        jc.set(qn("w:val"), "center")

        spacing = pPr.find(qn("w:spacing"))
        if spacing is None:
            spacing = ET.SubElement(pPr, qn("w:spacing"))
        spacing.set(qn("w:line"), "240")
        spacing.set(qn("w:lineRule"), "auto")
        spacing.set(qn("w:before"), "120")
        spacing.set(qn("w:after"), "0")

        # Remove first-line indent on image paragraphs
        ind = pPr.find(qn("w:ind"))
        if ind is not None:
            for attr in ("firstLine", "firstLineChars"):
                key = qn("w:" + attr)
                if key in ind.attrib:
                    ind.set(key, "0")

        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════
# 3. TABLE FORMATTING — 三线表
# ═══════════════════════════════════════════════════════════════════════

def format_tables(body) -> int:
    """Apply 三线表 style to all tables."""
    count = 0
    for tbl in body.findall(".//" + qn("w:tbl")):
        _format_table(tbl)
        count += 1
    return count


def _format_table(tbl):
    tbl_pr = tbl.find(qn("w:tblPr"))
    if tbl_pr is None:
        tbl_pr = ET.SubElement(tbl, qn("w:tblPr"))
        tbl.insert(0, tbl_pr)

    # Style: Table Grid (af7)
    ts = tbl_pr.find(qn("w:tblStyle"))
    if ts is None:
        ts = ET.SubElement(tbl_pr, qn("w:tblStyle"))
    ts.set(qn("w:val"), "af7")

    # Width: auto
    tw = tbl_pr.find(qn("w:tblW"))
    if tw is None:
        tw = ET.SubElement(tbl_pr, qn("w:tblW"))
    tw.set(qn("w:w"), "0")
    tw.set(qn("w:type"), "auto")

    # Remove fixed layout
    tl = tbl_pr.find(qn("w:tblLayout"))
    if tl is not None:
        tbl_pr.remove(tl)

    # Remove indent, use center alignment
    ti = tbl_pr.find(qn("w:tblInd"))
    if ti is not None:
        tbl_pr.remove(ti)
    jc = tbl_pr.find(qn("w:jc"))
    if jc is None:
        jc = ET.SubElement(tbl_pr, qn("w:jc"))
    jc.set(qn("w:val"), "center")

    # Borders: 三线表
    tb = tbl_pr.find(qn("w:tblBorders"))
    if tb is not None:
        tbl_pr.remove(tb)
    tb = ET.SubElement(tbl_pr, qn("w:tblBorders"))
    for name, val, sz in [
        ("top", "single", "12"), ("bottom", "single", "12"),
        ("left", "none", "0"), ("right", "none", "0"),
        ("insideH", "none", "0"), ("insideV", "none", "0"),
    ]:
        b = ET.SubElement(tb, qn("w:" + name))
        b.set(qn("w:val"), val)
        b.set(qn("w:sz"), sz)
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "auto")

    # tblLook
    look = tbl_pr.find(qn("w:tblLook"))
    if look is None:
        look = ET.SubElement(tbl_pr, qn("w:tblLook"))
    for k, v in [("val", "04A0"), ("firstRow", "1"), ("lastRow", "0"),
                 ("firstColumn", "1"), ("lastColumn", "0"),
                 ("noHBand", "0"), ("noVBand", "1")]:
        look.set(qn("w:" + k), v)

    # Format cells
    rows = tbl.findall(qn("w:tr"))
    for ri, row in enumerate(rows):
        is_hdr = (ri == 0)
        is_last = (ri == len(rows) - 1)
        for cell in row.findall(qn("w:tc")):
            _format_cell(cell, is_hdr, is_last)


def _format_cell(cell, is_hdr: bool, is_last: bool):
    tc_pr = cell.find(qn("w:tcPr"))
    if tc_pr is None:
        tc_pr = ET.SubElement(cell, qn("w:tcPr"))
        cell.insert(0, tc_pr)

    # Cell borders
    cb = tc_pr.find(qn("w:tcBorders"))
    if cb is not None:
        tc_pr.remove(cb)
    cb = ET.SubElement(tc_pr, qn("w:tcBorders"))

    if is_hdr:
        for name, sz in [("top", "12"), ("bottom", "4")]:
            b = ET.SubElement(cb, qn("w:" + name))
            b.set(qn("w:val"), "single"); b.set(qn("w:sz"), sz)
            b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto")
    elif is_last:
        b = ET.SubElement(cb, qn("w:bottom"))
        b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "12")
        b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto")

    for side in ("left", "right"):
        b = ET.SubElement(cb, qn("w:" + side))
        b.set(qn("w:val"), "none"); b.set(qn("w:sz"), "0")
        b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto")

    # Vertical center
    va = tc_pr.find(qn("w:vAlign"))
    if va is None:
        va = ET.SubElement(tc_pr, qn("w:vAlign"))
    va.set(qn("w:val"), "center")

    # Paragraph formatting inside cell
    for p in cell.findall(qn("w:p")):
        pPr = p.find(qn("w:pPr"))
        if pPr is None:
            pPr = ET.SubElement(p, qn("w:pPr"))
            p.insert(0, pPr)

        jc = pPr.find(qn("w:jc"))
        if jc is None:
            jc = ET.SubElement(pPr, qn("w:jc"))
        jc.set(qn("w:val"), "center")

        ind = pPr.find(qn("w:ind"))
        if ind is None:
            ind = ET.SubElement(pPr, qn("w:ind"))
        ind.set(qn("w:firstLineChars"), "0")
        ind.set(qn("w:firstLine"), "0")

        # Font size 10.5pt = 21 half-pt
        rPr = pPr.find(qn("w:rPr"))
        if rPr is None:
            rPr = ET.SubElement(pPr, qn("w:rPr"))
        for tag in ("w:sz", "w:szCs"):
            el = rPr.find(qn(tag))
            if el is None:
                el = ET.SubElement(rPr, qn(tag))
            el.set(qn("w:val"), "21")

        for r in p.findall(qn("w:r")):
            rPr_r = r.find(qn("w:rPr"))
            if rPr_r is None:
                rPr_r = ET.SubElement(r, qn("w:rPr"))
                r.insert(0, rPr_r)
            for tag in ("w:sz", "w:szCs"):
                el = rPr_r.find(qn(tag))
                if el is None:
                    el = ET.SubElement(rPr_r, qn(tag))
                el.set(qn("w:val"), "21")


# ═══════════════════════════════════════════════════════════════════════
# 4. PAGE HEADERS
# ═══════════════════════════════════════════════════════════════════════

def _make_header_xml(title: str) -> bytes:
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    xml += '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"\n'
    xml += '       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
    xml += '  <w:p>\n'
    xml += '    <w:pPr>\n'
    xml += '      <w:pStyle w:val="a4"/>\n'
    xml += '      <w:jc w:val="center"/>\n'
    xml += '    </w:pPr>\n'
    xml += '    <w:r>\n'
    xml += '      <w:rPr>\n'
    xml += '        <w:rFonts w:ascii="SimSun" w:hAnsi="SimSun" w:eastAsia="SimSun"/>\n'
    xml += '        <w:sz w:val="18"/>\n'
    xml += '        <w:szCs w:val="18"/>\n'
    xml += '      </w:rPr>\n'
    xml += '      <w:t xml:space="preserve">%s</w:t>\n' % title
    xml += '    </w:r>\n'
    xml += '  </w:p>\n'
    xml += '</w:hdr>\n'
    return xml.encode("utf-8")


def fix_headers(body, all_files: dict) -> int:
    """Update chapter header XML files based on Heading 1 positions."""
    children = list(body)

    # Find H1 paragraphs
    h1_list = []
    for i, elem in enumerate(children):
        if elem.tag == qn("w:p") and get_style(elem) == "1":
            h1_list.append(i)

    # Find section breaks with header references
    sect_breaks = {}
    for i, elem in enumerate(children):
        if elem.tag != qn("w:p"):
            continue
        pPr = elem.find(qn("w:pPr"))
        if pPr is None:
            continue
        sect = pPr.find(qn("w:sectPr"))
        if sect is None:
            continue
        for h in sect.findall(qn("w:headerReference")):
            if h.get(qn("w:type")) == "default":
                sect_breaks[i] = h.get(qn("r:id"))

    # Parse relationships
    rels_xml = all_files.get("word/_rels/document.xml.rels", b"")
    rels_root = ET.fromstring(rels_xml) if rels_xml else None
    rid_to_file = {}
    if rels_root is not None:
        for rel in rels_root:
            rid = rel.get("Id")
            target = rel.get("Target")
            if target and "header" in target.lower():
                rid_to_file[rid] = target

    updated = 0
    for ch_idx, h1_pos in enumerate(h1_list):
        if ch_idx >= len(CHAPTER_HEADERS):
            break
        title = CHAPTER_HEADERS[ch_idx]
        for offset in range(1, 5):
            check = h1_pos - offset
            if check in sect_breaks:
                rid = sect_breaks[check]
                if rid in rid_to_file:
                    hfile = "word/" + rid_to_file[rid]
                    all_files[hfile] = _make_header_xml(title)
                    updated += 1
                break

    # Handle 参考文献 header after last chapter
    if h1_list:
        last_h1 = h1_list[-1]
        for pos in range(last_h1 + 1, min(last_h1 + 200, len(children))):
            if pos in sect_breaks:
                rid = sect_breaks[pos]
                if rid in rid_to_file:
                    hfile = "word/" + rid_to_file[rid]
                    all_files[hfile] = _make_header_xml("\u53c2\u8003\u6587\u732e")
                    updated += 1
                break

    return updated


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def process(docx_path: Path, output_path: Path, fix_page_headers: bool = True) -> dict:
    stats: Counter = Counter()

    # Read all files from zip
    with zipfile.ZipFile(docx_path, "r") as zin:
        all_files = {name: zin.read(name) for name in zin.namelist()}

    # Parse document.xml
    root = ET.fromstring(all_files["word/document.xml"])
    body = root.find(qn("w:body"))
    if body is None:
        raise RuntimeError("No w:body found")

    # 1. Style remapping + caption detection
    style_stats = remap_styles(body)
    stats.update(style_stats)

    # 2. Image paragraph formatting
    stats["image_para"] = format_image_paragraphs(body)

    # 3. Table formatting (三线表)
    stats["table_formatted"] = format_tables(body)

    # 4. Page headers
    if fix_page_headers:
        stats["headers_updated"] = fix_headers(body, all_files)

    # Serialize document.xml back
    all_files["word/document.xml"] = ET.tostring(
        root, encoding="unicode", xml_declaration=True
    ).encode("utf-8")

    # Write output
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_files.items():
            zout.writestr(name, data)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive thesis DOCX post-processor"
    )
    parser.add_argument("--docx", required=True, type=Path, help="Input DOCX")
    parser.add_argument("--output", type=Path, help="Output DOCX")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite input (creates .bak-* backup)")
    parser.add_argument("--no-headers", action="store_true",
                        help="Skip page header updates")
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

    if args.in_place:
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".docx"))
        shutil.copy2(args.docx, tmp)
        stats = process(tmp, output, fix_page_headers=not args.no_headers)
        tmp.unlink()
    else:
        stats = process(args.docx, output, fix_page_headers=not args.no_headers)

    print(f"output: {output}")
    print("post-processing stats:")
    for k, v in stats.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
