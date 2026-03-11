from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from docx_xml_utils import extract_body_children

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

CAPTION_RE = re.compile(
    r"^\s*(图|Figure)\s*([0-9]+(?:[.\-][0-9]+)*)"
    r"(?:\s+|[:：]+)(.+?)\s*$",
    re.IGNORECASE,
)
FIGURE_REF_RE = re.compile(r"(图|Figure)\s*([0-9]+(?:[.\-][0-9]+)*)", re.IGNORECASE)


@dataclass(frozen=True)
class FigureAsset:
    number: str
    caption: str
    target: str
    suffix: str


def _get_block_text(element: ET.Element) -> str:
    chunks: list[str] = []
    for node in element.iter():
        local = node.tag.split("}")[-1]
        if local == "t" and node.text:
            chunks.append(node.text)
        elif local == "tab":
            chunks.append("\t")
        elif local in {"br", "cr"}:
            chunks.append("\n")
    return "".join(chunks).strip()


def _normalize_figure_number(number: str) -> str:
    return re.sub(r"[.\uFF0E\u3002]+", "-", number.strip())


def extract_figure_number(text: str) -> str | None:
    match = FIGURE_REF_RE.search(text or "")
    if not match:
        return None
    return _normalize_figure_number(match.group(2))


def _extract_caption(text: str) -> tuple[str, str] | None:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    match = CAPTION_RE.match(compact)
    if not match:
        return None
    number = _normalize_figure_number(match.group(2))
    return number, compact


def _looks_like_figure_block(block: ET.Element, image_targets: list[str]) -> bool:
    if not image_targets:
        return False
    local = block.tag.split("}")[-1]
    text = _get_block_text(block)
    if local == "tbl":
        return True
    if not text:
        return True
    if len(text) <= 12 and len(image_targets) <= 2:
        return True
    return False


def _collect_image_targets(block: ET.Element, rel_map: dict[str, str]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for node in block.iter():
        local = node.tag.split("}")[-1]
        rel_id: str | None = None
        if local == "blip":
            rel_id = node.get(f"{{{R_NS}}}embed") or node.get(f"{{{R_NS}}}link")
        elif local == "imagedata":
            rel_id = node.get(f"{{{R_NS}}}id")
        if not rel_id:
            continue
        target = rel_map.get(rel_id)
        if not target or target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return targets


def _load_rel_map(docx_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(docx_path) as zf:
        rels_root = ET.fromstring(zf.read("word/_rels/document.xml.rels"))
    rel_map: dict[str, str] = {}
    for rel in rels_root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        rel_id = rel.get("Id")
        rel_type = rel.get("Type")
        target = rel.get("Target")
        if rel_id and rel_type == IMAGE_REL_TYPE and target:
            rel_map[rel_id] = target
    return rel_map


def _find_nearby_caption(children: list[ET.Element], index: int) -> tuple[str, str] | None:
    fallback_number: str | None = None
    fallback_title: str | None = None
    for offset in (1, -1, 2, -2, 3, -3):
        candidate_index = index + offset
        if candidate_index < 0 or candidate_index >= len(children):
            continue
        block_text = _get_block_text(children[candidate_index])
        candidate = _extract_caption(block_text)
        if candidate:
            return candidate
        number = extract_figure_number(block_text)
        if number and fallback_number is None:
            fallback_number = number
        compact = re.sub(r"\s+", " ", block_text).strip()
        if compact and len(compact) <= 80 and extract_figure_number(compact) is None and fallback_title is None:
            fallback_title = compact
    if fallback_number:
        if fallback_title:
            return fallback_number, f"图{fallback_number} {fallback_title}"
        return fallback_number, f"图{fallback_number}"
    return None


@lru_cache(maxsize=8)
def build_figure_catalog(docx_path_str: str) -> dict[str, FigureAsset]:
    docx_path = Path(docx_path_str)
    rel_map = _load_rel_map(docx_path)
    children = extract_body_children(docx_path)
    catalog: dict[str, FigureAsset] = {}
    with zipfile.ZipFile(docx_path) as zf:
        for index, block in enumerate(children):
            image_targets = _collect_image_targets(block, rel_map)
            if not _looks_like_figure_block(block, image_targets):
                continue
            caption = _find_nearby_caption(children, index)
            if not caption:
                continue
            number, caption_text = caption
            if number in catalog:
                continue
            chosen_target = image_targets[0]
            member = f"word/{chosen_target.lstrip('/')}"
            if member not in zf.namelist():
                continue
            suffix = Path(chosen_target).suffix.lower() or ".bin"
            catalog[number] = FigureAsset(
                number=number,
                caption=caption_text,
                target=chosen_target,
                suffix=suffix,
            )
    return catalog


def resolve_figure_asset(docx_path: Path, lookup_text: str) -> tuple[bytes, FigureAsset] | None:
    number = extract_figure_number(lookup_text)
    if not number:
        return None
    catalog = build_figure_catalog(str(docx_path.resolve()))
    asset = catalog.get(number)
    if not asset:
        return None
    member = f"word/{asset.target.lstrip('/')}"
    with zipfile.ZipFile(docx_path) as zf:
        return zf.read(member), asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List figure numbers discovered in a DOCX by pairing image blocks with nearby figure captions."
    )
    parser.add_argument("docx", type=Path, help="Source DOCX used as the figure catalog")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_figure_catalog(str(args.docx.resolve()))
    for number in sorted(catalog):
        asset = catalog[number]
        print(f"{asset.number}\t{asset.caption}\t{asset.target}")
    print(f"figures: {len(catalog)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
