# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from sync_markdown_to_docx import main as sync_main

# Sync Chapter 6
sys.argv = ['sync',
            '--docx', 'D:/桌面/lunwen/thesis/writing/versions/test.docx',
            '--markdown', 'D:/桌面/lunwen/thesis/writing/第6章_总结与展望_final.md',
            '--match-heading', '总结与展望',
            '--level', '1',
            '--output', 'D:/桌面/lunwen/thesis/writing/versions/test.docx']

print("Syncing Chapter 6...")
sync_main()
print("Chapter 6 done!")
