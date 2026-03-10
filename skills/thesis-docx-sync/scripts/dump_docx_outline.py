from __future__ import annotations

import argparse
from pathlib import Path

from docx_xml_utils import build_outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump heading outline from a DOCX file for section-level syncing."
    )
    parser.add_argument("docx", type=Path, help="Path to the target .docx file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.docx.suffix.lower() != ".docx":
        raise SystemExit("Only .docx is supported. Convert legacy .doc first.")

    outline = build_outline(args.docx)
    for item in outline:
        indent = "  " * (int(item["level"]) - 1)
        print(
            f"{item['level']}\t{item['body_index']}\t{item['style_name']}\t"
            f"{indent}{item['text']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
