from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Markdown section files into one chapter-level Markdown file "
            "for DOCX sync workflows."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output Markdown path. Keep it near the source files if they use relative images.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Chapter title inserted as the top-level heading.",
    )
    parser.add_argument(
        "--shift-headings",
        type=int,
        default=1,
        help="How many heading levels to shift down inside each input file. Default: 1.",
    )
    parser.add_argument(
        "--replace-text",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Literal text rewrite applied after heading shift. Can be repeated.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Markdown files to merge in order.",
    )
    return parser.parse_args()


def parse_replacements(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --replace-text value: {item}")
        old, new = item.split("=", 1)
        if not old:
            raise SystemExit(f"Invalid --replace-text value: {item}")
        pairs.append((old, new))
    return pairs


def shift_headings(text: str, amount: int) -> str:
    if amount <= 0:
        return text

    lines = text.splitlines()
    in_fence = False
    fence_marker = ""
    shifted_lines: list[str] = []

    for line in lines:
        fence_match = re.match(r"^([`~]{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            shifted_lines.append(line)
            continue

        if not in_fence:
            heading_match = re.match(r"^(#{1,6})(\s+.*)$", line)
            if heading_match:
                hashes = heading_match.group(1)
                rest = heading_match.group(2)
                new_level = min(len(hashes) + amount, 6)
                shifted_lines.append("#" * new_level + rest)
                continue

        shifted_lines.append(line)

    return "\n".join(shifted_lines)


def main() -> int:
    args = parse_args()
    replacements = parse_replacements(args.replace_text)

    sections: list[str] = []
    for markdown in args.inputs:
        text = markdown.read_text(encoding="utf-8")
        text = shift_headings(text, args.shift_headings).strip()
        for old, new in replacements:
            text = text.replace(old, new)
        sections.append(text)

    merged = "# " + args.title.strip() + "\n\n" + "\n\n".join(sections).strip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(merged, encoding="utf-8", newline="\n")

    print(f"output: {args.output}")
    print(f"chapter title: {args.title}")
    print(f"merged inputs: {len(args.inputs)}")
    for markdown in args.inputs:
        print(f"  - {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
