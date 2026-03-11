from __future__ import annotations

import argparse
import copy
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from docx_xml_utils import NS, PKG_REL, find_section_range, get_body, get_document_root, qn


CAPTIONS = [
    "图 3-1 CST 仿真模型",
    "图 3-2 等离子体模型设置界面图",
    "图 3-3 MATLAB 理论计算的 Drude 模型时延曲线",
    "图 3-4 固定碰撞频率条件下不同电子密度的群时延曲线（CST 全波仿真）",
    "图 3-5 固定截止频率条件下不同碰撞频率的群时延曲线（CST 全波仿真）",
    "图 3-6 低电子密度区间 CST 仿真群时延曲线",
    "图 3-7 高电子密度区间 CST 仿真群时延曲线",
    "图 3-8 不同参数组合下群时延曲线的多解性交点",
    "图 3-9 差频信号瞬时频率演化的时频分析对比",
    "图 3-10 不同色散强度下差频信号的 FFT 频谱特征",
    "图 3-11 带宽-散焦非线性耦合关系",
    "图 3-12 展宽零点示意",
    "图 3-13 色散效应工程判据的参数空间约束",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".emf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center chapter-3 figures, add visible captions, and normalize spacing."
    )
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _get_relationship_map(docx_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(docx_path) as zf:
        rels_root = ET.fromstring(zf.read("word/_rels/document.xml.rels"))
    return {
        rel.get("Id"): rel.get("Target", "")
        for rel in rels_root.findall("pr:Relationship", PKG_REL)
    }


def _get_caption_style_id(docx_path: Path) -> str | None:
    with zipfile.ZipFile(docx_path) as zf:
        styles_root = ET.fromstring(zf.read("word/styles.xml"))
    for style in styles_root.findall("w:style", NS):
        name_el = style.find("w:name", NS)
        name = name_el.get(qn("w:val"), "").strip().lower() if name_el is not None else ""
        if name == "caption":
            return style.get(qn("w:styleId"))
    return None


def _get_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == qn("w:t") and node.text:
            parts.append(node.text)
    return "".join(parts).strip()


def _ensure_ppr(paragraph: ET.Element) -> ET.Element:
    ppr = paragraph.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(qn("w:pPr"))
        paragraph.insert(0, ppr)
    return ppr


def _set_alignment(paragraph: ET.Element, value: str) -> None:
    ppr = _ensure_ppr(paragraph)
    jc = ppr.find("w:jc", NS)
    if jc is None:
        jc = ET.SubElement(ppr, qn("w:jc"))
    jc.set(qn("w:val"), value)


def _set_spacing(paragraph: ET.Element, before: int, after: int) -> None:
    ppr = _ensure_ppr(paragraph)
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.SubElement(ppr, qn("w:spacing"))
    spacing.set(qn("w:before"), str(before))
    spacing.set(qn("w:after"), str(after))
    spacing.set(qn("w:line"), "360")
    spacing.set(qn("w:lineRule"), "auto")


def _set_style(paragraph: ET.Element, style_id: str | None) -> None:
    if not style_id:
        return
    ppr = _ensure_ppr(paragraph)
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        pstyle = ET.SubElement(ppr, qn("w:pStyle"))
    pstyle.set(qn("w:val"), style_id)


def _set_keep_next(paragraph: ET.Element, enabled: bool) -> None:
    ppr = _ensure_ppr(paragraph)
    node = ppr.find("w:keepNext", NS)
    if enabled:
        if node is None:
            ET.SubElement(ppr, qn("w:keepNext"))
    elif node is not None:
        ppr.remove(node)


def _create_caption_paragraph(text: str, style_id: str | None) -> ET.Element:
    paragraph = ET.Element(qn("w:p"))
    _set_style(paragraph, style_id)
    _set_alignment(paragraph, "center")
    _set_spacing(paragraph, before=0, after=120)

    run = ET.SubElement(paragraph, qn("w:r"))
    rpr = ET.SubElement(run, qn("w:rPr"))
    ET.SubElement(rpr, qn("w:noProof"))
    t = ET.SubElement(run, qn("w:t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return paragraph


def _normalize_caption_paragraph(paragraph: ET.Element, caption: str, style_id: str | None) -> None:
    for child in list(paragraph):
        paragraph.remove(child)

    _set_style(paragraph, style_id)
    _set_alignment(paragraph, "center")
    _set_spacing(paragraph, before=0, after=120)
    _set_keep_next(paragraph, False)

    run = ET.SubElement(paragraph, qn("w:r"))
    rpr = ET.SubElement(run, qn("w:rPr"))
    ET.SubElement(rpr, qn("w:noProof"))
    t = ET.SubElement(run, qn("w:t"))
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = caption


def _paragraph_is_figure(paragraph: ET.Element, rel_map: dict[str, str]) -> bool:
    if _get_paragraph_text(paragraph):
        return False
    drawings = paragraph.findall(".//w:drawing", NS)
    if len(drawings) != 1:
        return False

    blip = drawings[0].find(".//a:blip", NS)
    if blip is None:
        return False
    rid = blip.get(qn("r:embed"))
    target = rel_map.get(rid or "", "")
    return Path(target).suffix.lower() in IMAGE_EXTENSIONS


def _find_figure_block_indexes(
    body_children: list[ET.Element], start_index: int, end_index: int, rel_map: dict[str, str]
) -> list[int]:
    indexes: list[int] = []
    for index in range(start_index + 1, end_index):
        block = body_children[index]
        if block.tag != qn("w:p"):
            continue
        if _paragraph_is_figure(block, rel_map):
            indexes.append(index)
    return indexes


def main() -> None:
    args = parse_args()
    if not args.docx.exists():
        raise SystemExit(f"DOCX not found: {args.docx}")

    start_index, end_index, _ = find_section_range(
        args.docx, "宽带信号在色散介质中的传播机理与误差量化", level=1
    )
    rel_map = _get_relationship_map(args.docx)
    caption_style_id = _get_caption_style_id(args.docx)

    root = get_document_root(args.docx)
    body = get_body(root)
    children = list(body)
    figure_indexes = _find_figure_block_indexes(children, start_index, end_index, rel_map)

    if len(figure_indexes) < len(CAPTIONS):
        raise SystemExit(
            f"Expected at least {len(CAPTIONS)} figure paragraphs in chapter 3, found {len(figure_indexes)}."
        )

    figure_indexes = figure_indexes[: len(CAPTIONS)]

    for body_index in figure_indexes:
        paragraph = children[body_index]
        _set_alignment(paragraph, "center")
        _set_spacing(paragraph, before=120, after=0)
        _set_keep_next(paragraph, True)

    for body_index, caption in reversed(list(zip(figure_indexes, CAPTIONS))):
        next_block = children[body_index + 1] if body_index + 1 < len(children) else None
        if next_block is not None and next_block.tag == qn("w:p"):
            next_text = _get_paragraph_text(next_block)
            if next_text == caption:
                _normalize_caption_paragraph(next_block, caption, caption_style_id)
                continue

        caption_paragraph = _create_caption_paragraph(caption, caption_style_id)
        body.insert(body_index + 1, caption_paragraph)

    output_path = args.output
    with zipfile.ZipFile(args.docx) as source_zip, zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as target_zip:
        document_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        for info in source_zip.infolist():
            data = document_bytes if info.filename == "word/document.xml" else source_zip.read(info.filename)
            target_zip.writestr(info, data)


if __name__ == "__main__":
    main()
