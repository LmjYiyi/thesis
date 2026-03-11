from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


INLINE_MATH_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_MATH_RISK_TOKENS = (
    r"\frac",
    r"\dfrac",
    r"\tfrac",
    r"\underbrace",
    r"\overbrace",
    r"\left",
    r"\right",
    r"\begin{",
)


@dataclass
class Finding:
    severity: str
    path: Path
    line_no: int
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight Markdown before DOCX sync. "
            "Checks image paths and inline math patterns that commonly render poorly in Word."
        )
    )
    parser.add_argument(
        "--markdown",
        nargs="+",
        required=True,
        type=Path,
        help="Markdown file(s) to inspect.",
    )
    parser.add_argument(
        "--allowed-image-prefix",
        action="append",
        default=[],
        help=(
            "Allowed local image path prefix, e.g. figures/final_output_doc/. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--forbid-image-basename-prefix",
        action="append",
        default=[],
        help="Forbid image basenames that start with these prefixes. Can be repeated.",
    )
    parser.add_argument(
        "--inline-math-max-length",
        type=int,
        default=60,
        help="Warn when inline math exceeds this many characters. Default: 60.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when warnings are found.",
    )
    parser.add_argument(
        "--profile",
        choices=("final_output_doc",),
        help=(
            "Apply a project profile. 'final_output_doc' enforces images from "
            "figures/final_output_doc/ and forbids sync/preview helper assets."
        ),
    )
    return parser.parse_args()


def _normalize_path_text(raw_path: str) -> str:
    return raw_path.replace("\\", "/").strip()


def _apply_profile(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    allowed_prefixes = list(args.allowed_image_prefix)
    forbidden_prefixes = list(args.forbid_image_basename_prefix)

    if args.profile == "final_output_doc":
        for item in (
            "figures/final_output_doc/",
            "writing/figures/final_output_doc/",
        ):
            if item not in allowed_prefixes:
                allowed_prefixes.append(item)
        for item in ("sync_fig", "_preview_"):
            if item not in forbidden_prefixes:
                forbidden_prefixes.append(item)

    normalized_allowed = [
        item.replace("\\", "/").lstrip("./") for item in allowed_prefixes if item.strip()
    ]
    normalized_forbidden = [item.strip() for item in forbidden_prefixes if item.strip()]
    return normalized_allowed, normalized_forbidden


def _is_allowed_local_image(raw_path: str, allowed_prefixes: list[str]) -> bool:
    if not allowed_prefixes:
        return True
    normalized = _normalize_path_text(raw_path).lstrip("./")
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)


def _scan_markdown(
    markdown_path: Path,
    allowed_prefixes: list[str],
    forbidden_basename_prefixes: list[str],
    inline_math_max_length: int,
) -> list[Finding]:
    findings: list[Finding] = []
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    fence_marker = ""

    for idx, line in enumerate(lines, start=1):
        fence_match = re.match(r"^([`~]{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if in_fence:
            continue

        for _, raw_path in IMAGE_PATTERN.findall(line):
            normalized = _normalize_path_text(raw_path)

            if normalized.startswith("docx-figure://"):
                findings.append(
                    Finding(
                        "warning",
                        markdown_path,
                        idx,
                        f"Found DOCX-backed image placeholder '{raw_path}'. "
                        "Prefer a concrete local image path in the canonical figure directory.",
                    )
                )
                continue

            if re.match(r"^[a-zA-Z]+://", normalized):
                findings.append(
                    Finding(
                        "warning",
                        markdown_path,
                        idx,
                        f"Found external image URL '{raw_path}'. Prefer local figure assets for thesis sync.",
                    )
                )
                continue

            basename = Path(normalized).name
            if any(basename.startswith(prefix) for prefix in forbidden_basename_prefixes):
                findings.append(
                    Finding(
                        "error",
                        markdown_path,
                        idx,
                        f"Image '{raw_path}' uses a forbidden helper basename. "
                        "Do not leave sync/preview artifacts in the thesis figure library.",
                    )
                )

            if not _is_allowed_local_image(normalized, allowed_prefixes):
                allowed_text = ", ".join(allowed_prefixes) or "<any local path>"
                findings.append(
                    Finding(
                        "error",
                        markdown_path,
                        idx,
                        f"Image '{raw_path}' is outside the allowed figure directory. Allowed prefixes: {allowed_text}.",
                    )
                )

        if "$$" in line:
            continue

        for match in INLINE_MATH_PATTERN.finditer(line):
            math_text = match.group(1).strip()
            compact_length = len(re.sub(r"\s+", "", math_text))

            if any(token in math_text for token in INLINE_MATH_RISK_TOKENS):
                findings.append(
                    Finding(
                        "warning",
                        markdown_path,
                        idx,
                        f"Inline math '{math_text[:80]}' contains display-style constructs. "
                        "Prefer slash notation in body text, or move it to display math.",
                    )
                )
                continue

            if compact_length > inline_math_max_length:
                findings.append(
                    Finding(
                        "warning",
                        markdown_path,
                        idx,
                        f"Inline math is too long ({compact_length} chars). "
                        "Shorten it, flatten it, or move it to display math for Word readability.",
                    )
                )
            elif "/" in math_text and compact_length > 40:
                findings.append(
                    Finding(
                        "warning",
                        markdown_path,
                        idx,
                        f"Inline math '{math_text[:80]}' is still visually dense. "
                        "Consider shortening it or moving it to display math.",
                    )
                )

    return findings


def main() -> int:
    args = parse_args()
    allowed_prefixes, forbidden_basename_prefixes = _apply_profile(args)

    all_findings: list[Finding] = []
    for markdown_path in args.markdown:
        if not markdown_path.exists():
            raise SystemExit(f"Markdown file not found: {markdown_path}")
        all_findings.extend(
            _scan_markdown(
                markdown_path,
                allowed_prefixes,
                forbidden_basename_prefixes,
                args.inline_math_max_length,
            )
        )

    errors = [item for item in all_findings if item.severity == "error"]
    warnings = [item for item in all_findings if item.severity == "warning"]

    if all_findings:
        for item in all_findings:
            print(f"{item.severity.upper()}: {item.path}:{item.line_no}: {item.message}")
    else:
        print("OK: no image-path or inline-math readability issues found.")

    if errors:
        return 1
    if warnings and args.fail_on_warning:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
