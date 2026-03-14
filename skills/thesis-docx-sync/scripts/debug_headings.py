# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from docx_xml_utils import build_outline, normalize_heading_text
import json

# Build outline
items = build_outline('D:/桌面/lunwen/thesis/writing/versions/test.docx')

# Get all level 1 headings
print("=== Level 1 headings in test.docx ===")
for item in items:
    if item['level'] == '1':
        print(f"Body index: {item['body_index']}")
        print(f"  Text (raw repr): {repr(item['text'])}")
        print(f"  Normalized: {item['normalized']}")
        print()

# Now try to match
target = "等离子体电磁特性与LFMCW诊断机理"
normalized_target = normalize_heading_text(target)
print(f"=== Target heading ===")
print(f"Original: {target}")
print(f"Normalized: {normalized_target}")
print()

# Check if any heading matches
print("=== Matching ===")
for item in items:
    if item['level'] == '1':
        if item['normalized'] == normalized_target:
            print(f"EXACT MATCH at body_index {item['body_index']}")
        # Check partial
        if normalized_target in item['normalized'] or item['normalized'] in normalized_target:
            print(f"PARTIAL MATCH at body_index {item['body_index']}: {item['normalized']}")
