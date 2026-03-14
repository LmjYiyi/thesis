# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from sync_markdown_to_docx import main as sync_main
import argparse

# Create args
args = argparse.Namespace(
    docx='D:/桌面/lunwen/thesis/writing/versions/test.docx',
    markdown='D:/桌面/lunwen/thesis/writing/第2章_等离子体电磁特性与LFMCW诊断机理_final.md',
    match_heading='等离子体电磁特性与LFMCW诊断机理',
    level=1,
    reference_docx=None,
    output='D:/桌面/lunwen/thesis/writing/versions/test.docx',
    in_place=False,
    missing_images='error',
    image_source_docx=None
)

# Override sys.argv
original_argv = sys.argv
sys.argv = ['sync', '--docx', args.docx, '--markdown', args.markdown,
            '--match-heading', args.match_heading, '--level', str(args.level),
            '--output', args.output]

try:
    sync_main()
except SystemExit:
    pass
finally:
    sys.argv = original_argv
