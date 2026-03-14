# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'D:/桌面/lunwen/thesis/skills/thesis-docx-sync/scripts')

from sync_markdown_to_docx import main as sync_main

# Override sys.argv
sys.argv = ['sync',
            '--docx', 'D:/桌面/lunwen/thesis/writing/versions/test.docx',
            '--markdown', 'D:/桌面/lunwen/thesis/writing/第2章_等离子体电磁特性与LFMCW诊断机理_final.md',
            '--match-heading', '等离子体诊断原理及线性调频连续波诊断原理',
            '--level', '1',
            '--output', 'D:/桌面/lunwen/thesis/writing/versions/test.docx']

sync_main()
