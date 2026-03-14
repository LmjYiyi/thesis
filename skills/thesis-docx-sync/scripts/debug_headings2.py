# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from docx_xml_utils import build_outline, normalize_heading_text
import json

# Build outline
items = build_outline('D:/桌面/lunwen/thesis/writing/versions/test.docx')

# Write to file
with open('D:/桌面/lunwen/thesis/writing/versions/debug_headings.json', 'w', encoding='utf-8') as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Total items: {len(items)}")
print(f"Written to debug_headings.json")
