from __future__ import annotations

import argparse
import copy
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from docx_xml_utils import NS, get_body, get_document_root, qn


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".emf"}
CAPTION_TITLES = [
    "CST 仿真模型",
    "等离子体模型设置界面图",
    "MATLAB 理论计算的 Drude 模型时延曲线",
    "固定碰撞频率条件下不同电子密度的群时延曲线（CST 全波仿真）",
    "固定截止频率条件下不同碰撞频率的群时延曲线（CST 全波仿真）",
    "低电子密度区间 CST 仿真群时延曲线",
    "高电子密度区间 CST 仿真群时延曲线",
    "不同参数组合下群时延曲线的多解性交点",
    "差频信号瞬时频率演化的时频分析对比",
    "不同色散强度下差频信号的 FFT 频谱特征",
    "带宽-散焦非线性耦合关系",
    "展宽零点示意",
    "色散效应工程判据的参数空间约束",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite chapter-3 figure and caption paragraphs to match the reference thesis DOCX format."
    )
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-docx", type=Path)
    return parser.parse_args()


def _get_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == qn("w:t") and node.text:
            parts.append(node.text)
    return "".join(parts).strip()


def _rel_map(docx_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(docx_path) as zf:
        rels_root = ET.fromstring(zf.read("word/_rels/document.xml.rels"))
    return {
        rel.get("Id"): rel.get("Target", "")
        for rel in rels_root.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    }


def _find_reference_docx(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    matches = sorted(Path("writing").glob("*终稿*曹杠.docx"))
    if not matches:
        raise SystemExit("Reference DOCX not found under writing\\*.docx")
    return matches[0]


def _load_reference_templates(reference_docx: Path) -> tuple[ET.Element, ET.Element]:
    root = get_document_root(reference_docx)
    body = get_body(root)
    paragraphs = body.findall("w:p", NS)

    image_template: ET.Element | None = None
    caption_template: ET.Element | None = None

    for idx, paragraph in enumerate(paragraphs):
        text = _get_paragraph_text(paragraph)
        drawings = paragraph.findall(".//w:drawing", NS)
        if image_template is None and len(drawings) == 1 and not text:
            image_template = copy.deepcopy(paragraph)
            if idx + 1 < len(paragraphs):
                next_paragraph = paragraphs[idx + 1]
                pstyle = next_paragraph.find("w:pPr/w:pStyle", NS)
                if pstyle is not None and pstyle.get(qn("w:val")) == "-0":
                    caption_template = copy.deepcopy(next_paragraph)
            if image_template is not None and caption_template is not None:
                break

    if image_template is None or caption_template is None:
        raise SystemExit("Failed to extract image/caption paragraph templates from reference DOCX.")
    return image_template, caption_template


def _find_chapter_range(body: ET.Element) -> tuple[int, int]:
    children = list(body)
    start = None
    end = len(children)
    for idx, child in enumerate(children):
        if child.tag != qn("w:p"):
            continue
        text = _get_paragraph_text(child)
        if text == "宽带信号在色散介质中的传播机理与误差量化":
            start = idx
            continue
        if start is not None and text in {"系统差频信号数据处理", "系统标定及诊断实验", "总结与展望"}:
            end = idx
            break
    if start is None:
        raise SystemExit("Chapter 3 heading not found in target DOCX.")
    return start, end


def _is_image_only_paragraph(paragraph: ET.Element, rel_map: dict[str, str]) -> bool:
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


def _find_figure_indexes(
    children: list[ET.Element], start: int, end: int, rel_map: dict[str, str]
) -> list[int]:
    indexes: list[int] = []
    for idx in range(start + 1, end):
        child = children[idx]
        if child.tag != qn("w:p"):
            continue
        if _is_image_only_paragraph(child, rel_map):
            indexes.append(idx)
    return indexes


def _ensure_use_local_dpi(blip: ET.Element) -> None:
    extlst = blip.find("a:extLst", NS)
    if extlst is None:
        extlst = ET.SubElement(blip, qn("a:extLst"))
    for ext in extlst.findall("a:ext", NS):
        if ext.get("uri") == "{28A0092B-C50C-407E-A947-70E740481C1C}":
            use = ext.find("{http://schemas.microsoft.com/office/drawing/2010/main}useLocalDpi")
            if use is None:
                use = ET.SubElement(ext, "{http://schemas.microsoft.com/office/drawing/2010/main}useLocalDpi")
            use.set("val", "0")
            return
    ext = ET.SubElement(extlst, qn("a:ext"))
    ext.set("uri", "{28A0092B-C50C-407E-A947-70E740481C1C}")
    use = ET.SubElement(ext, "{http://schemas.microsoft.com/office/drawing/2010/main}useLocalDpi")
    use.set("val", "0")


def _normalize_inline_drawing(inline: ET.Element, frame_template: ET.Element) -> None:
    for attr in ("distT", "distB", "distL", "distR"):
        inline.set(attr, "0")

    effect = inline.find("wp:effectExtent", NS)
    if effect is None:
        effect = ET.Element(qn("wp:effectExtent"))
        effect.set("l", "0")
        effect.set("t", "0")
        effect.set("r", "0")
        effect.set("b", "0")
        extent = inline.find("wp:extent", NS)
        insert_pos = 1 if extent is not None else 0
        inline.insert(insert_pos, effect)
    else:
        effect.set("l", "0")
        effect.set("t", "0")
        effect.set("r", "0")
        effect.set("b", "0")

    docpr = inline.find("wp:docPr", NS)
    pic_nvpr = inline.find(".//pic:cNvPr", NS)
    pic_id = "1"
    if docpr is not None and docpr.get("id"):
        pic_id = docpr.get("id")
    elif pic_nvpr is not None and pic_nvpr.get("id"):
        pic_id = pic_nvpr.get("id")

    if docpr is not None:
        for attr in ("descr", "title"):
            if attr in docpr.attrib:
                del docpr.attrib[attr]
        docpr.set("name", f"图片 {pic_id}")
        docpr.set("id", pic_id)

    if inline.find("wp:cNvGraphicFramePr", NS) is None:
        graphic = inline.find("a:graphic", NS)
        insert_pos = list(inline).index(graphic) if graphic is not None else len(list(inline))
        inline.insert(insert_pos, copy.deepcopy(frame_template))

    if pic_nvpr is not None:
        if "descr" in pic_nvpr.attrib:
            del pic_nvpr.attrib["descr"]
        pic_nvpr.set("name", f"图片 {pic_id}")
        pic_nvpr.set("id", pic_id)

    blip = inline.find(".//a:blip", NS)
    if blip is not None:
        blip.set("cstate", "hqprint")
        _ensure_use_local_dpi(blip)

    sppr = inline.find(".//pic:spPr", NS)
    if sppr is not None:
        for attr in list(sppr.attrib):
            del sppr.attrib[attr]
        ln = sppr.find("a:ln", NS)
        if ln is None:
            ln = ET.SubElement(sppr, qn("a:ln"))
        for child in list(ln):
            ln.remove(child)
        ET.SubElement(ln, qn("a:noFill"))


def _rewrite_image_paragraph(paragraph: ET.Element, image_template: ET.Element, frame_template: ET.Element) -> None:
    current_drawing = paragraph.find(".//w:drawing", NS)
    if current_drawing is None:
        return

    paragraph[:] = []

    template_ppr = image_template.find("w:pPr", NS)
    if template_ppr is not None:
        paragraph.append(copy.deepcopy(template_ppr))

    template_run = image_template.find("w:r", NS)
    if template_run is None:
        return
    new_run = copy.deepcopy(template_run)
    for child in list(new_run):
        if child.tag == qn("w:drawing"):
            new_run.remove(child)
    new_run.append(copy.deepcopy(current_drawing))
    paragraph.append(new_run)

    inline = paragraph.find(".//wp:inline", NS)
    if inline is not None:
        _normalize_inline_drawing(inline, frame_template)


def _rewrite_caption_paragraph(paragraph: ET.Element, caption_template: ET.Element, title: str) -> None:
    paragraph[:] = []
    template_ppr = caption_template.find("w:pPr", NS)
    if template_ppr is not None:
        paragraph.append(copy.deepcopy(template_ppr))

    run_space = ET.SubElement(paragraph, qn("w:r"))
    rpr_space = ET.SubElement(run_space, qn("w:rPr"))
    rfonts_space = ET.SubElement(rpr_space, qn("w:rFonts"))
    rfonts_space.set(qn("w:hint"), "eastAsia")
    t_space = ET.SubElement(run_space, qn("w:t"))
    t_space.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t_space.text = " "

    run_text = ET.SubElement(paragraph, qn("w:r"))
    rpr_text = ET.SubElement(run_text, qn("w:rPr"))
    rfonts_text = ET.SubElement(rpr_text, qn("w:rFonts"))
    rfonts_text.set(qn("w:hint"), "eastAsia")
    t_text = ET.SubElement(run_text, qn("w:t"))
    t_text.text = title


def main() -> None:
    args = parse_args()
    reference_docx = _find_reference_docx(args.reference_docx)
    image_template, caption_template = _load_reference_templates(reference_docx)
    frame_template = image_template.find(".//wp:cNvGraphicFramePr", NS)
    if frame_template is None:
        raise SystemExit("Reference image template is missing wp:cNvGraphicFramePr.")

    rel_map = _rel_map(args.docx)
    root = get_document_root(args.docx)
    body = get_body(root)
    start, end = _find_chapter_range(body)
    children = list(body)
    figure_indexes = _find_figure_indexes(children, start, end, rel_map)

    if len(figure_indexes) < len(CAPTION_TITLES):
        raise SystemExit(
            f"Expected at least {len(CAPTION_TITLES)} figure paragraphs in chapter 3, found {len(figure_indexes)}."
        )

    figure_indexes = figure_indexes[: len(CAPTION_TITLES)]

    for idx, title in zip(figure_indexes, CAPTION_TITLES):
        image_paragraph = children[idx]
        _rewrite_image_paragraph(image_paragraph, image_template, frame_template)

        next_idx = idx + 1
        if next_idx < len(children) and children[next_idx].tag == qn("w:p"):
            caption_paragraph = children[next_idx]
            _rewrite_caption_paragraph(caption_paragraph, caption_template, title)
        else:
            caption_paragraph = ET.Element(qn("w:p"))
            _rewrite_caption_paragraph(caption_paragraph, caption_template, title)
            body.insert(next_idx, caption_paragraph)

    with zipfile.ZipFile(args.docx) as source_zip, zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED
    ) as target_zip:
        document_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        for info in source_zip.infolist():
            data = document_bytes if info.filename == "word/document.xml" else source_zip.read(info.filename)
            target_zip.writestr(info, data)


if __name__ == "__main__":
    main()
