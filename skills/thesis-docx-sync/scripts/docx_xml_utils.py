from __future__ import annotations

import copy
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
V_NS = "urn:schemas-microsoft-com:vml"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W15_NS = "http://schemas.microsoft.com/office/word/2012/wordml"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
HYPERLINK_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
FOOTNOTES_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
ENDNOTES_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/endnotes"
FOOTNOTES_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"
ENDNOTES_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml"

NS = {
    "w": W_NS,
    "r": R_NS,
    "m": M_NS,
    "a": A_NS,
    "pic": PIC_NS,
    "wp": WP_NS,
    "v": V_NS,
    "mc": MC_NS,
    "w14": W14_NS,
    "w15": W15_NS,
}
PKG_REL = {"pr": PKG_REL_NS}
CT = {"ct": CONTENT_TYPES_NS}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
ET.register_namespace("", PKG_REL_NS)
ET.register_namespace("", CONTENT_TYPES_NS)


def qn(name: str) -> str:
    prefix, local = name.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def read_xml_from_docx(docx_path: Path, member: str) -> ET.Element:
    with zipfile.ZipFile(docx_path) as zf:
        return ET.fromstring(zf.read(member))


def read_optional_xml_from_docx(docx_path: Path, member: str) -> ET.Element | None:
    with zipfile.ZipFile(docx_path) as zf:
        if member not in zf.namelist():
            return None
        return ET.fromstring(zf.read(member))


def load_style_map(docx_path: Path) -> dict[str, str]:
    styles_root = read_xml_from_docx(docx_path, "word/styles.xml")
    style_map: dict[str, str] = {}
    for style in styles_root.findall("w:style", NS):
        style_id = style.get(qn("w:styleId"))
        if not style_id:
            continue
        name_el = style.find("w:name", NS)
        style_map[style_id] = name_el.get(qn("w:val")) if name_el is not None else style_id
    return style_map


def get_document_root(docx_path: Path) -> ET.Element:
    return read_xml_from_docx(docx_path, "word/document.xml")


def get_body(root: ET.Element) -> ET.Element:
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("word/document.xml does not contain w:body")
    return body


def get_paragraph_text(paragraph: ET.Element) -> str:
    texts = []
    for node in paragraph.iter():
        if node.tag == qn("w:t") and node.text:
            texts.append(node.text)
        elif node.tag == qn("w:tab"):
            texts.append("\t")
        elif node.tag in {qn("w:br"), qn("w:cr")}:
            texts.append("\n")
    return "".join(texts).strip()


def get_paragraph_style_id(paragraph: ET.Element) -> str | None:
    ppr = paragraph.find("w:pPr", NS)
    if ppr is None:
        return None
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        return None
    return pstyle.get(qn("w:val"))


def detect_heading_level(paragraph: ET.Element, style_map: dict[str, str]) -> int | None:
    style_id = get_paragraph_style_id(paragraph)
    style_name = style_map.get(style_id or "", "")

    for candidate in (style_id or "", style_name):
        normalized = candidate.replace(" ", "").lower()
        if normalized in {"-1", "标题-无编号", "title-without-number"}:
            return 0

    for candidate in (style_id or "", style_name):
        match = re.search(r"heading\s*([1-9])", candidate, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"标题\s*([1-9])", candidate)
        if match:
            return int(match.group(1))

    ppr = paragraph.find("w:pPr", NS)
    if ppr is not None:
        outline = ppr.find("w:outlineLvl", NS)
        if outline is not None:
            value = outline.get(qn("w:val"))
            if value and value.isdigit():
                return int(value) + 1

    if style_id and style_id.isdigit():
        level = int(style_id)
        if 1 <= level <= 9:
            return level
    return None


def normalize_heading_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("“", "\"").replace("”", "\"")
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"^第[一二三四五六七八九十百零〇0-9]+章", "", normalized)
    normalized = re.sub(r"^[0-9]+(?:\.[0-9]+)*", "", normalized)
    return normalized.casefold()


def build_outline(docx_path: Path) -> list[dict[str, object]]:
    root = get_document_root(docx_path)
    body = get_body(root)
    style_map = load_style_map(docx_path)
    outline: list[dict[str, object]] = []
    for body_index, child in enumerate(list(body)):
        if child.tag != qn("w:p"):
            continue
        level = detect_heading_level(child, style_map)
        if level is None:
            continue
        text = get_paragraph_text(child)
        if not text:
            continue
        style_id = get_paragraph_style_id(child)
        outline.append(
            {
                "body_index": body_index,
                "level": level,
                "text": text,
                "normalized": normalize_heading_text(text),
                "style_id": style_id,
                "style_name": style_map.get(style_id or "", style_id or ""),
            }
        )
    return outline


def find_heading(docx_path: Path, heading_text: str, level: int | None = None) -> dict[str, object]:
    normalized_target = normalize_heading_text(heading_text)
    candidates = [
        item
        for item in build_outline(docx_path)
        if item["normalized"] == normalized_target and (level is None or item["level"] == level)
    ]
    if not candidates:
        raise ValueError(f"Heading not found: {heading_text}")
    if len(candidates) > 1:
        joined = "\n".join(
            f"- level {item['level']}: {item['text']}" for item in candidates[:10]
        )
        raise ValueError(
            "Heading is ambiguous. Pass --level to disambiguate.\n"
            f"Candidates:\n{joined}"
        )
    return candidates[0]


def find_section_range(docx_path: Path, heading_text: str, level: int | None = None) -> tuple[int, int, int]:
    root = get_document_root(docx_path)
    body = get_body(root)
    blocks = list(body)
    heading = find_heading(docx_path, heading_text, level=level)
    start_index = int(heading["body_index"])
    start_level = int(heading["level"])
    style_map = load_style_map(docx_path)

    end_index = len(blocks)
    for index in range(start_index + 1, len(blocks)):
        child = blocks[index]
        if child.tag == qn("w:sectPr"):
            end_index = index
            break
        if child.tag != qn("w:p"):
            continue
        child_level = detect_heading_level(child, style_map)
        if child_level is not None and child_level <= start_level:
            end_index = index
            break
    return start_index, end_index, start_level


def extract_body_children(docx_path: Path) -> list[ET.Element]:
    root = get_document_root(docx_path)
    body = get_body(root)
    return [copy.deepcopy(child) for child in list(body) if child.tag != qn("w:sectPr")]


def set_first_paragraph_style(elements: Iterable[ET.Element], style_id: str | None) -> None:
    if not style_id:
        return

    for element in elements:
        if element.tag != qn("w:p"):
            continue
        if not get_paragraph_text(element):
            continue

        ppr = element.find("w:pPr", NS)
        if ppr is None:
            ppr = ET.Element(qn("w:pPr"))
            element.insert(0, ppr)

        pstyle = ppr.find("w:pStyle", NS)
        if pstyle is None:
            pstyle = ET.Element(qn("w:pStyle"))
            ppr.insert(0, pstyle)

        pstyle.set(qn("w:val"), style_id)
        return


def scan_unsupported_features(elements: Iterable[ET.Element]) -> list[str]:
    seen: set[str] = set()
    for element in elements:
        for node in element.iter():
            tag = node.tag
            if tag == qn("w:object"):
                seen.add("embedded OLE objects")
            elif tag == qn("w:commentReference"):
                seen.add("comments")
            elif tag == qn("w:altChunk"):
                seen.add("altChunk content")
    return sorted(seen)


def _serialize_xml(root: ET.Element) -> bytes:
    if root.tag == f"{{{PKG_REL_NS}}}Relationships":
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
        data = data.replace("<ns0:Relationships", "<Relationships", 1)
        data = data.replace("</ns0:Relationships>", "</Relationships>")
        data = data.replace("xmlns:ns0=", "xmlns=", 1)
        data = data.replace("<ns0:Relationship", "<Relationship")
        return data.encode("utf-8")
    if root.tag == f"{{{CONTENT_TYPES_NS}}}Types":
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
        data = data.replace("<ns0:Types", "<Types", 1)
        data = data.replace("</ns0:Types>", "</Types>")
        data = data.replace("xmlns:ns0=", "xmlns=", 1)
        data = data.replace("<ns0:Default", "<Default")
        data = data.replace("<ns0:Override", "<Override")
        return data.encode("utf-8")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _list_zip_members(docx_path: Path) -> set[str]:
    with zipfile.ZipFile(docx_path) as zf:
        return {item.filename for item in zf.infolist()}


def _load_or_create_relationships(docx_path: Path, member: str) -> ET.Element:
    root = read_optional_xml_from_docx(docx_path, member)
    if root is not None:
        return root
    return ET.Element(f"{{{PKG_REL_NS}}}Relationships")


def _load_content_types(docx_path: Path) -> ET.Element:
    return read_xml_from_docx(docx_path, "[Content_Types].xml")


def _ensure_override(content_types_root: ET.Element, part_name: str, content_type: str) -> None:
    for item in content_types_root.findall("ct:Override", CT):
        if item.get("PartName") == part_name:
            item.set("ContentType", content_type)
            return
    node = ET.SubElement(content_types_root, f"{{{CONTENT_TYPES_NS}}}Override")
    node.set("PartName", part_name)
    node.set("ContentType", content_type)


def _collect_relationship_ids(elements: Iterable[ET.Element]) -> set[str]:
    rel_ids: set[str] = set()
    for element in elements:
        for node in element.iter():
            for attr_name, attr_value in node.attrib.items():
                if attr_name.startswith(f"{{{R_NS}}}") and attr_value:
                    rel_ids.add(attr_value)
    return rel_ids


def _replace_relationship_ids(elements: Iterable[ET.Element], rel_map: dict[str, str]) -> None:
    if not rel_map:
        return
    for element in elements:
        for node in element.iter():
            for attr_name, attr_value in list(node.attrib.items()):
                if attr_name.startswith(f"{{{R_NS}}}") and attr_value in rel_map:
                    node.set(attr_name, rel_map[attr_value])


def _next_rel_id(relationships_root: ET.Element) -> str:
    existing_ids = {
        rel.get("Id")
        for rel in relationships_root.findall("pr:Relationship", PKG_REL)
        if rel.get("Id")
    }
    index = 1
    while f"rId{index}" in existing_ids:
        index += 1
    return f"rId{index}"


def _copy_document_relationships(
    target_docx: Path,
    source_docx: Path,
    relationships_root: ET.Element,
    content_types_root: ET.Element,
    replacement_children: list[ET.Element],
    extra_members: dict[str, bytes],
) -> None:
    source_rels = read_optional_xml_from_docx(source_docx, "word/_rels/document.xml.rels")
    if source_rels is None:
        return

    source_rel_map = {
        rel.get("Id"): rel
        for rel in source_rels.findall("pr:Relationship", PKG_REL)
        if rel.get("Id")
    }
    target_members = _list_zip_members(target_docx) | set(extra_members)
    target_default_exts = {
        item.get("Extension", "").lower()
        for item in content_types_root.findall("ct:Default", CT)
    }
    source_defaults = {
        item.get("Extension", "").lower(): item.get("ContentType", "")
        for item in _load_content_types(source_docx).findall("ct:Default", CT)
    }

    rel_id_map: dict[str, str] = {}
    with zipfile.ZipFile(source_docx) as source_zip:
        for old_rel_id in sorted(_collect_relationship_ids(replacement_children)):
            rel = source_rel_map.get(old_rel_id)
            if rel is None:
                continue

            rel_type = rel.get("Type", "")
            if rel_type not in {IMAGE_REL_TYPE, HYPERLINK_REL_TYPE}:
                continue

            new_rel_id = _next_rel_id(relationships_root)
            new_rel = ET.SubElement(relationships_root, f"{{{PKG_REL_NS}}}Relationship")
            new_rel.set("Id", new_rel_id)
            new_rel.set("Type", rel_type)
            rel_id_map[old_rel_id] = new_rel_id

            target_mode = rel.get("TargetMode")
            if target_mode:
                new_rel.set("TargetMode", target_mode)

            source_target = rel.get("Target", "")
            if rel_type == HYPERLINK_REL_TYPE or target_mode == "External":
                new_rel.set("Target", source_target)
                continue

            source_member = posixpath.normpath(posixpath.join("word", source_target))
            suffix = Path(source_target).suffix.lower()
            stem = Path(source_target).stem or new_rel_id
            counter = 1
            while True:
                candidate_name = f"thesis-docx-sync-{stem}-{counter}{suffix}"
                candidate_member = f"word/media/{candidate_name}"
                if candidate_member not in target_members:
                    break
                counter += 1
            new_rel.set("Target", f"media/{candidate_name}")
            extra_members[candidate_member] = source_zip.read(source_member)
            target_members.add(candidate_member)

            extension = suffix.lstrip(".")
            if extension and extension not in target_default_exts and extension in source_defaults:
                node = ET.SubElement(content_types_root, f"{{{CONTENT_TYPES_NS}}}Default")
                node.set("Extension", extension)
                node.set("ContentType", source_defaults[extension])
                target_default_exts.add(extension)

    _replace_relationship_ids(replacement_children, rel_id_map)


def _max_numeric_id(roots: Iterable[ET.Element | None], tag_names: set[str], attr_name: str) -> int:
    max_id = 0
    for root in roots:
        if root is None:
            continue
        for node in root.iter():
            if node.tag not in tag_names:
                continue
            value = node.get(attr_name)
            if value and value.lstrip("-").isdigit():
                max_id = max(max_id, int(value))
    return max_id


def _remap_bookmark_ids(elements: Iterable[ET.Element], existing_max_id: int) -> int:
    next_id = existing_max_id + 1
    id_map: dict[str, str] = {}
    for element in elements:
        for node in element.iter():
            if node.tag == qn("w:bookmarkStart"):
                old_id = node.get(qn("w:id"))
                if not old_id:
                    continue
                new_id = str(next_id)
                next_id += 1
                id_map[old_id] = new_id
                node.set(qn("w:id"), new_id)
            elif node.tag == qn("w:bookmarkEnd"):
                old_id = node.get(qn("w:id"))
                if old_id in id_map:
                    node.set(qn("w:id"), id_map[old_id])
    return next_id - 1


def _remap_drawing_ids(elements: Iterable[ET.Element], existing_max_id: int) -> int:
    next_id = existing_max_id + 1
    for element in elements:
        for node in element.iter():
            if node.tag not in {qn("wp:docPr"), qn("pic:cNvPr")}:
                continue
            value = node.get("id")
            if value and value.isdigit():
                node.set("id", str(next_id))
                next_id += 1
    return next_id - 1


def _collect_note_reference_ids(elements: Iterable[ET.Element], ref_tag: str) -> set[str]:
    ids: set[str] = set()
    for element in elements:
        for node in element.iter():
            if node.tag == ref_tag:
                note_id = node.get(qn("w:id"))
                if note_id and note_id.lstrip("-").isdigit():
                    ids.add(note_id)
    return ids


def _replace_note_reference_ids(elements: Iterable[ET.Element], ref_tag: str, id_map: dict[str, str]) -> None:
    for element in elements:
        for node in element.iter():
            if node.tag != ref_tag:
                continue
            old_id = node.get(qn("w:id"))
            if old_id in id_map:
                node.set(qn("w:id"), id_map[old_id])


def _ensure_document_part_relationship(
    relationships_root: ET.Element,
    target: str,
    rel_type: str,
) -> None:
    for rel in relationships_root.findall("pr:Relationship", PKG_REL):
        if rel.get("Type") == rel_type:
            rel.set("Target", target)
            return
    new_rel = ET.SubElement(relationships_root, f"{{{PKG_REL_NS}}}Relationship")
    new_rel.set("Id", _next_rel_id(relationships_root))
    new_rel.set("Type", rel_type)
    new_rel.set("Target", target)


def _merge_note_part(
    target_docx: Path,
    source_docx: Path,
    relationships_root: ET.Element,
    content_types_root: ET.Element,
    replacement_children: list[ET.Element],
    note_kind: str,
) -> tuple[tuple[str, ET.Element] | None, list[ET.Element]]:
    config = {
        "footnotes": {
            "part_name": "word/footnotes.xml",
            "ref_tag": qn("w:footnoteReference"),
            "note_tag": qn("w:footnote"),
            "rel_type": FOOTNOTES_REL_TYPE,
            "content_type": FOOTNOTES_CONTENT_TYPE,
            "target": "footnotes.xml",
        },
        "endnotes": {
            "part_name": "word/endnotes.xml",
            "ref_tag": qn("w:endnoteReference"),
            "note_tag": qn("w:endnote"),
            "rel_type": ENDNOTES_REL_TYPE,
            "content_type": ENDNOTES_CONTENT_TYPE,
            "target": "endnotes.xml",
        },
    }[note_kind]

    referenced_ids = _collect_note_reference_ids(replacement_children, config["ref_tag"])
    if not referenced_ids:
        return None, []

    source_root = read_optional_xml_from_docx(source_docx, config["part_name"])
    if source_root is None:
        raise ValueError(f"Source DOCX does not contain {config['part_name']}")

    target_root = read_optional_xml_from_docx(target_docx, config["part_name"])
    if target_root is None:
        target_root = ET.Element(source_root.tag, source_root.attrib)
        for child in list(source_root):
            if child.get(qn("w:type")) is not None:
                target_root.append(copy.deepcopy(child))

    existing_ids = {
        int(node.get(qn("w:id")))
        for node in target_root.findall(f"w:{note_kind[:-1]}", NS)
        if node.get(qn("w:id")) and node.get(qn("w:id")).isdigit()
    }
    next_note_id = max(existing_ids | {0}) + 1

    source_note_map = {
        node.get(qn("w:id")): node
        for node in source_root.findall(f"w:{note_kind[:-1]}", NS)
        if node.get(qn("w:id")) is not None
    }

    id_map: dict[str, str] = {}
    copied_notes: list[ET.Element] = []
    for old_id in sorted(referenced_ids, key=int):
        source_note = source_note_map.get(old_id)
        if source_note is None:
            raise ValueError(f"Referenced {note_kind[:-1]} id {old_id} not found in source DOCX")
        note_copy = copy.deepcopy(source_note)
        new_id = str(next_note_id)
        next_note_id += 1
        note_copy.set(qn("w:id"), new_id)
        copied_notes.append(note_copy)
        id_map[old_id] = new_id

    _replace_note_reference_ids(replacement_children, config["ref_tag"], id_map)
    for note in copied_notes:
        target_root.append(note)

    _ensure_document_part_relationship(relationships_root, config["target"], config["rel_type"])
    _ensure_override(content_types_root, f"/{config['part_name']}", config["content_type"])
    return (config["part_name"], target_root), copied_notes


def replace_body_range(
    docx_path: Path,
    source_docx_path: Path,
    output_path: Path,
    start_index: int,
    end_index: int,
    replacement_children: list[ET.Element],
) -> None:
    target_root = get_document_root(docx_path)
    target_body = get_body(target_root)
    original_children = list(target_body)

    content_types_root = _load_content_types(docx_path)
    document_rels_root = _load_or_create_relationships(docx_path, "word/_rels/document.xml.rels")
    extra_members: dict[str, bytes] = {}
    extra_xml_parts: dict[str, ET.Element] = {}

    footnotes_root = read_optional_xml_from_docx(docx_path, "word/footnotes.xml")
    endnotes_root = read_optional_xml_from_docx(docx_path, "word/endnotes.xml")

    max_bookmark_id = _max_numeric_id(
        [target_root, footnotes_root, endnotes_root],
        {qn("w:bookmarkStart"), qn("w:bookmarkEnd")},
        qn("w:id"),
    )
    max_drawing_id = _max_numeric_id(
        [target_root, footnotes_root, endnotes_root],
        {qn("wp:docPr"), qn("pic:cNvPr")},
        "id",
    )

    max_bookmark_id = _remap_bookmark_ids(replacement_children, max_bookmark_id)
    max_drawing_id = _remap_drawing_ids(replacement_children, max_drawing_id)

    note_parts: list[tuple[str, ET.Element] | None] = []
    note_copies: list[list[ET.Element]] = []
    for note_kind in ("footnotes", "endnotes"):
        part_info, copied_notes = _merge_note_part(
            docx_path,
            source_docx_path,
            document_rels_root,
            content_types_root,
            replacement_children,
            note_kind,
        )
        note_parts.append(part_info)
        note_copies.append(copied_notes)

    for copied_notes in note_copies:
        if copied_notes:
            max_bookmark_id = _remap_bookmark_ids(copied_notes, max_bookmark_id)
            max_drawing_id = _remap_drawing_ids(copied_notes, max_drawing_id)

    _copy_document_relationships(
        docx_path,
        source_docx_path,
        document_rels_root,
        content_types_root,
        replacement_children,
        extra_members,
    )

    for part_info in note_parts:
        if part_info is not None:
            part_name, root = part_info
            extra_xml_parts[part_name] = root

    for child in original_children[start_index:end_index]:
        target_body.remove(child)

    insert_at = start_index
    for child in replacement_children:
        target_body.insert(insert_at, copy.deepcopy(child))
        insert_at += 1

    document_bytes = _serialize_xml(target_root)
    rel_bytes = _serialize_xml(document_rels_root)
    content_types_bytes = _serialize_xml(content_types_root)

    temp_output = output_path.with_suffix(output_path.suffix + ".tmp") if output_path == docx_path else output_path
    with zipfile.ZipFile(docx_path) as source_zip, zipfile.ZipFile(
        temp_output, "w", compression=zipfile.ZIP_DEFLATED
    ) as target_zip:
        for info in source_zip.infolist():
            if info.filename == "word/document.xml":
                data = document_bytes
            elif info.filename == "word/_rels/document.xml.rels":
                data = rel_bytes
            elif info.filename == "[Content_Types].xml":
                data = content_types_bytes
            elif info.filename in extra_xml_parts:
                data = _serialize_xml(extra_xml_parts[info.filename])
            else:
                data = source_zip.read(info.filename)
            target_zip.writestr(info, data)

        for part_name, root in extra_xml_parts.items():
            if part_name not in source_zip.namelist():
                target_zip.writestr(part_name, _serialize_xml(root))

        for member_name, member_bytes in extra_members.items():
            target_zip.writestr(member_name, member_bytes)

    if temp_output != output_path:
        temp_output.replace(output_path)
