from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from docx_figure_catalog import resolve_figure_asset
from docx_xml_utils import (
    extract_body_children,
    find_heading,
    find_section_range,
    get_paragraph_style_id,
    get_paragraph_text,
    qn,
    replace_body_range,
    scan_unsupported_features,
    set_first_paragraph_style,
)

EQNO_MARKER = "THESIS_DOCX_SYNC_EQNO:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Markdown to DOCX with pandoc, then replace one heading-bounded "
            "section inside a thesis DOCX."
        )
    )
    parser.add_argument("--docx", required=True, type=Path, help="Target thesis .docx file")
    parser.add_argument("--markdown", required=True, type=Path, help="Source Markdown file")
    parser.add_argument(
        "--match-heading",
        required=True,
        help="Existing heading text inside the target DOCX used as the replacement anchor",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=range(1, 10),
        help="Heading level used to disambiguate duplicate titles",
    )
    parser.add_argument(
        "--reference-docx",
        type=Path,
        help="Reference DOCX passed to pandoc. Defaults to --docx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output DOCX path. Defaults to <target>_synced.docx unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the target DOCX after creating a timestamped backup next to it.",
    )
    parser.add_argument(
        "--missing-images",
        choices=("error", "placeholder"),
        default="error",
        help=(
            "How to handle Markdown images whose files cannot be resolved. "
            "Default: error."
        ),
    )
    parser.add_argument(
        "--image-source-docx",
        type=Path,
        help=(
            "Optional source DOCX used as a fallback figure library when a Markdown image "
            "path cannot be resolved locally. Matching is done by figure number such as 图5-1."
        ),
    )
    return parser.parse_args()


def ensure_supported_input(args: argparse.Namespace) -> None:
    if args.docx.suffix.lower() != ".docx":
        raise SystemExit("Only .docx is supported. Convert legacy .doc first.")
    if args.reference_docx and args.reference_docx.suffix.lower() != ".docx":
        raise SystemExit("--reference-docx must point to a .docx file.")
    if args.image_source_docx and args.image_source_docx.suffix.lower() != ".docx":
        raise SystemExit("--image-source-docx must point to a .docx file.")
    if not args.docx.exists():
        raise SystemExit(f"Target DOCX not found: {args.docx}")
    if not args.markdown.exists():
        raise SystemExit(f"Markdown file not found: {args.markdown}")
    if args.image_source_docx and not args.image_source_docx.exists():
        raise SystemExit(f"Image source DOCX not found: {args.image_source_docx}")


def _rewrite_tagged_display_math(text: str) -> str:
    def extract_braced(source: str, open_index: int) -> tuple[str, int]:
        if open_index >= len(source) or source[open_index] != "{":
            raise ValueError("Expected braced group.")

        depth = 0
        chunks: list[str] = []
        index = open_index
        while index < len(source):
            char = source[index]
            if char == "{":
                depth += 1
                if depth > 1:
                    chunks.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chunks), index + 1
                chunks.append(char)
            else:
                chunks.append(char)
            index += 1
        raise ValueError("Unclosed braced group.")

    def strip_underbrace_annotations(source: str) -> str:
        output: list[str] = []
        index = 0
        token = r"\underbrace"

        while index < len(source):
            if source.startswith(token, index):
                brace_index = index + len(token)
                while brace_index < len(source) and source[brace_index].isspace():
                    brace_index += 1
                if brace_index >= len(source) or source[brace_index] != "{":
                    output.append(source[index])
                    index += 1
                    continue

                expr, next_index = extract_braced(source, brace_index)
                scan = next_index
                while scan < len(source) and source[scan].isspace():
                    scan += 1
                if scan < len(source) and source[scan] == "_":
                    scan += 1
                    while scan < len(source) and source[scan].isspace():
                        scan += 1
                    if scan < len(source) and source[scan] == "{":
                        _, next_index = extract_braced(source, scan)

                output.append(strip_underbrace_annotations(expr))
                index = next_index
                continue

            output.append(source[index])
            index += 1

        return "".join(output)

    def normalize_math_for_mathtype(source: str) -> str:
        source = strip_underbrace_annotations(source)
        source = source.replace(r"\qquad", " ")
        source = source.replace(r"\quad", " ")
        source = re.sub(r"\s+", " ", source)
        return source.strip()

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        tag_match = re.search(r"\\tag\{([^{}]+)\}", body)
        if not tag_match:
            normalized = normalize_math_for_mathtype(body)
            return f"$${normalized}$$"
        tag_value = tag_match.group(1).strip()
        rendered_tag = tag_value if tag_value.startswith("(") else f"({tag_value})"
        rewritten = re.sub(r"\s*\\tag\{[^{}]+\}\s*", " ", body, count=1).strip()
        rewritten = normalize_math_for_mathtype(rewritten)
        return f"$${rewritten}$$\n\n{EQNO_MARKER}{rendered_tag}\n"

    return re.sub(r"\$\$(.*?)\$\$", repl, text, flags=re.DOTALL)


def _ensure_paragraph_alignment(paragraph, alignment: str) -> None:
    ppr = paragraph.find(qn("w:pPr"))
    if ppr is None:
        ppr = ET.Element(qn("w:pPr"))
        paragraph.insert(0, ppr)

    jc = ppr.find(qn("w:jc"))
    if jc is None:
        jc = ET.SubElement(ppr, qn("w:jc"))
    jc.set(qn("w:val"), alignment)


def _set_paragraph_style(paragraph, style_id: str | None) -> None:
    if not style_id:
        return
    ppr = paragraph.find(qn("w:pPr"))
    if ppr is None:
        ppr = ET.Element(qn("w:pPr"))
        paragraph.insert(0, ppr)

    pstyle = ppr.find(qn("w:pStyle"))
    if pstyle is None:
        pstyle = ET.SubElement(ppr, qn("w:pStyle"))
    pstyle.set(qn("w:val"), style_id)


def _ensure_mt_display_tabs(paragraph) -> None:
    ppr = paragraph.find(qn("w:pPr"))
    if ppr is None:
        ppr = ET.Element(qn("w:pPr"))
        paragraph.insert(0, ppr)

    tabs = ppr.find(qn("w:tabs"))
    if tabs is not None:
        ppr.remove(tabs)
    tabs = ET.SubElement(ppr, qn("w:tabs"))

    center_tab = ET.SubElement(tabs, qn("w:tab"))
    center_tab.set(qn("w:val"), "center")
    center_tab.set(qn("w:pos"), "4540")

    right_tab = ET.SubElement(tabs, qn("w:tab"))
    right_tab.set(qn("w:val"), "right")
    right_tab.set(qn("w:pos"), "9080")


def _ensure_display_equation_spacing(paragraph) -> None:
    ppr = paragraph.find(qn("w:pPr"))
    if ppr is None:
        ppr = ET.Element(qn("w:pPr"))
        paragraph.insert(0, ppr)

    spacing = ppr.find(qn("w:spacing"))
    if spacing is None:
        spacing = ET.SubElement(ppr, qn("w:spacing"))
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")
    spacing.set(qn("w:line"), "400")
    spacing.set(qn("w:lineRule"), "atLeast")


def _append_tab_run(paragraph) -> None:
    run = ET.SubElement(paragraph, qn("w:r"))
    ET.SubElement(run, qn("w:tab"))


def _create_spacer_paragraph(style_id: str | None):
    paragraph = ET.Element(qn("w:p"))
    _set_paragraph_style(paragraph, style_id)
    run = ET.SubElement(paragraph, qn("w:r"))
    text = ET.SubElement(run, qn("w:t"))
    text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text.text = " "
    return paragraph


def _append_equation_number(paragraph, equation_number: str) -> None:
    for part in re.split(r"(-)", equation_number):
        run = ET.SubElement(paragraph, qn("w:r"))
        if part == "-":
            ET.SubElement(run, qn("w:noBreakHyphen"))
            continue
        text = ET.SubElement(run, qn("w:t"))
        text.text = part


def _build_tagged_equation_paragraph(equation_paragraph, equation_number: str):
    paragraph = ET.Element(qn("w:p"))
    _set_paragraph_style(paragraph, "MTDisplayEquation")
    _ensure_mt_display_tabs(paragraph)
    _ensure_display_equation_spacing(paragraph)

    _append_tab_run(paragraph)

    omath_para = equation_paragraph.find(qn("m:oMathPara"))
    omath = omath_para.find(qn("m:oMath")) if omath_para is not None else None
    if omath is None:
        omath = equation_paragraph.find(qn("m:oMath"))
    if omath is None:
        return equation_paragraph
    paragraph.append(copy.deepcopy(omath))

    _append_tab_run(paragraph)
    _append_equation_number(paragraph, equation_number)
    return paragraph


def _extract_equation_number_marker(paragraph) -> str | None:
    text = get_paragraph_text(paragraph)
    if not text.startswith(EQNO_MARKER):
        return None
    marker = text[len(EQNO_MARKER):].strip()
    return marker or None


def postprocess_rendered_children(children: list) -> list:
    rewritten = []
    index = 0
    while index < len(children):
        current = children[index]
        if (
            current.tag == qn("w:p")
            and current.find(f".//{qn('m:oMathPara')}") is not None
            and index + 1 < len(children)
            and children[index + 1].tag == qn("w:p")
        ):
            equation_number = _extract_equation_number_marker(children[index + 1])
            if equation_number:
                rewritten.append(_create_spacer_paragraph("a0"))
                rewritten.append(_build_tagged_equation_paragraph(current, equation_number))
                rewritten.append(_create_spacer_paragraph("a0"))
                index += 2
                continue

        if current.tag == qn("w:p") and _extract_equation_number_marker(current):
            index += 1
            continue

        rewritten.append(current)
        index += 1

    return rewritten


def _strip_heading_number_prefixes(text: str) -> str:
    heading_pattern = re.compile(r"^(#{1,6})([ \t]+)(.+)$", re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        marker = match.group(1)
        spacing = match.group(2)
        heading_text = match.group(3).strip()

        normalized = re.sub(
            r"^第[一二三四五六七八九十百零〇0-9]+章[：:、.\s-]*",
            "",
            heading_text,
        )
        normalized = re.sub(
            r"^[0-9]+(?:\.[0-9]+)*[：:、.\s-]+",
            "",
            normalized,
        )
        normalized = normalized.strip()
        if not normalized:
            normalized = heading_text
        return f"{marker}{spacing}{normalized}"

    return heading_pattern.sub(repl, text)


def _resolve_image_source(image_path: str, markdown_path: Path, project_root: Path) -> Path | None:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = (markdown_path.parent / image_path).resolve()
    if direct.exists():
        return direct

    from_root = (project_root / image_path).resolve()
    if from_root.exists():
        return from_root

    basename = Path(image_path).name
    if not basename:
        return None

    matches = sorted(project_root.rglob(basename))
    if len(matches) == 1:
        return matches[0]

    normalized_tail = image_path.replace("\\", "/").lstrip("./")
    tail_matches = [
        item for item in matches if item.as_posix().endswith(normalized_tail)
    ]
    if len(tail_matches) == 1:
        return tail_matches[0]
    return None


def _build_missing_image_placeholder(alt_text: str, raw_path: str) -> str:
    label = alt_text.strip() or Path(raw_path).name or raw_path
    return f"[Image missing: {label} | source: {raw_path}]"


def _stage_docx_figure_asset(
    alt_text: str,
    raw_path: str,
    image_source_docx: Path | None,
    staged_dir: Path,
    staged_index: int,
) -> tuple[str | None, int]:
    if image_source_docx is None:
        return None, staged_index

    lookup_text = alt_text.strip()
    if raw_path.startswith("docx-figure://"):
        lookup_text = raw_path[len("docx-figure://") :].strip() or lookup_text
    elif not lookup_text:
        lookup_text = Path(raw_path).name

    resolved = resolve_figure_asset(image_source_docx, lookup_text)
    if resolved is None:
        return None, staged_index

    payload, asset = resolved
    staged_name = f"image-{staged_index}{asset.suffix}"
    staged_path = staged_dir / staged_name
    staged_path.write_bytes(payload)
    rendered_alt = alt_text or asset.caption
    return f"![{rendered_alt}](media_assets/{staged_name})", staged_index + 1


def _stage_markdown_images(
    text: str,
    markdown_path: Path,
    temp_dir: Path,
    missing_image_mode: str,
    image_source_docx: Path | None,
) -> tuple[str, list[str]]:
    project_root = Path.cwd().resolve()
    staged_dir = temp_dir / "media_assets"
    staged_dir.mkdir(exist_ok=True)
    missing: list[str] = []
    staged_index = 1

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

    def repl(match: re.Match[str]) -> str:
        nonlocal staged_index
        alt_text = match.group(1)
        raw_path = match.group(2)
        if re.match(r"^[a-zA-Z]+://", raw_path) and not raw_path.startswith("docx-figure://"):
            return match.group(0)

        resolved = _resolve_image_source(raw_path, markdown_path, project_root)
        if resolved is None:
            staged_markdown, next_index = _stage_docx_figure_asset(
                alt_text,
                raw_path,
                image_source_docx,
                staged_dir,
                staged_index,
            )
            if staged_markdown is not None:
                staged_index = next_index
                return staged_markdown
        if resolved is None:
            missing.append(raw_path)
            if missing_image_mode == "placeholder":
                return _build_missing_image_placeholder(alt_text, raw_path)
            return match.group(0)

        staged_name = f"image-{staged_index}{resolved.suffix.lower()}"
        staged_index += 1
        staged_path = staged_dir / staged_name
        shutil.copy2(resolved, staged_path)
        return f"![{alt_text}](media_assets/{staged_name})"

    return image_pattern.sub(repl, text), missing


def preprocess_markdown(
    markdown: Path,
    temp_dir: Path,
    missing_image_mode: str,
    image_source_docx: Path | None,
) -> tuple[Path, list[str]]:
    original_text = markdown.read_text(encoding="utf-8")
    rewritten_text = _rewrite_tagged_display_math(original_text)
    rewritten_text = _strip_heading_number_prefixes(rewritten_text)
    rewritten_text, missing_images = _stage_markdown_images(
        rewritten_text,
        markdown,
        temp_dir,
        missing_image_mode,
        image_source_docx,
    )
    if missing_images:
        unique = ", ".join(sorted(set(missing_images)))
        if missing_image_mode == "error":
            raise SystemExit(f"Could not resolve Markdown image paths: {unique}")

    processed_markdown = temp_dir / markdown.name
    processed_markdown.write_text(rewritten_text, encoding="utf-8", newline="\n")
    return processed_markdown, sorted(set(missing_images))


def render_markdown(
    markdown: Path,
    reference_docx: Path,
    temp_dir: Path,
    missing_image_mode: str,
    image_source_docx: Path | None,
) -> tuple[Path, list[str]]:
    processed_markdown, missing_images = preprocess_markdown(
        markdown,
        temp_dir,
        missing_image_mode,
        image_source_docx,
    )
    rendered_docx = temp_dir / "rendered.docx"
    resource_path = ";".join(
        [
            str(processed_markdown.parent.resolve()),
            str(markdown.parent.resolve()),
            str(Path.cwd().resolve()),
        ]
    )
    cmd = [
        "pandoc",
        processed_markdown.name,
        f"--reference-doc={reference_docx.resolve()}",
        f"--resource-path={resource_path}",
        "-o",
        str(rendered_docx),
    ]
    subprocess.run(cmd, check=True, cwd=processed_markdown.parent)
    return rendered_docx, missing_images


def resolve_output_path(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.in_place:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.docx.with_suffix(f".bak-{timestamp}.docx")
        shutil.copy2(args.docx, backup)
        return args.docx, backup

    if args.output:
        return args.output, None

    return args.docx.with_name(f"{args.docx.stem}_synced.docx"), None


def main() -> int:
    args = parse_args()
    ensure_supported_input(args)

    reference_docx = args.reference_docx or args.docx
    output_path, backup_path = resolve_output_path(args)

    with tempfile.TemporaryDirectory(prefix="thesis-docx-sync-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        rendered_docx, missing_images = render_markdown(
            args.markdown,
            reference_docx,
            temp_dir,
            args.missing_images,
            args.image_source_docx,
        )
        source_children = extract_body_children(rendered_docx)
        source_children = postprocess_rendered_children(source_children)
        issues = scan_unsupported_features(source_children)
        if issues:
            joined = ", ".join(issues)
            raise SystemExit(
                "Rendered Markdown contains unsupported Word features: "
                f"{joined}."
            )

        matched_heading = find_heading(
            args.docx,
            args.match_heading,
            level=args.level,
        )
        start_index, end_index, detected_level = find_section_range(
            args.docx,
            args.match_heading,
            level=args.level,
        )
        set_first_paragraph_style(
            source_children,
            matched_heading.get("style_id"),
        )
        try:
            replace_body_range(
                args.docx,
                rendered_docx,
                output_path,
                start_index,
                end_index,
                source_children,
            )
        except PermissionError as exc:
            raise SystemExit(
                f"Output DOCX is locked or open: {output_path}. "
                "Close it in Word or choose another --output path."
            ) from exc

    print(f"target: {args.docx}")
    if backup_path:
        print(f"backup: {backup_path}")
    print(f"output: {output_path}")
    print(
        f"replaced heading: {args.match_heading} "
        f"(level {args.level or detected_level}, body blocks {start_index}:{end_index})"
    )
    print(f"source markdown: {args.markdown}")
    if missing_images:
        print(
            "missing images downgraded to placeholders: "
            + ", ".join(missing_images)
        )
    if args.image_source_docx:
        print(f"image source docx: {args.image_source_docx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
