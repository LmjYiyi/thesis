# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from sync_markdown_to_docx import main as sync_main

# Sync Chapter 4
sys.argv = ['sync',
            '--docx', 'D:/桌面/lunwen/thesis/writing/versions/test.docx',
            '--markdown', 'D:/桌面/lunwen/thesis/writing/_chapter4_sync_tmp.md',
            '--match-heading', '系统差频信号数据处理',
            '--level', '1',
            '--output', 'D:/桌面/lunwen/thesis/writing/versions/test.docx']

print("Syncing Chapter 4...")
sync_main()
print("Chapter 4 done!")
