import re

with open('D:/projects/wukong_ai/pathfinding/behavior_clone_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add 'import gc' after 'import argparse'
old_import = 'import os\nimport sys\nimport argparse\nimport numpy as np'
new_import = 'import os\nimport sys\nimport argparse\nimport gc\nimport numpy as np'
if old_import in content:
    content = content.replace(old_import, new_import)
    print('Fix 1 applied: added import gc')
else:
    print('Fix 1 FAILED')

# Fix 2: Add memory cleanup after epoch
old_text = '        if acc > best_acc:'
new_text = '        # Memory cleanup after each epoch\n        gc.collect()\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n\n        if acc > best_acc:'
if old_text in content:
    content = content.replace(old_text, new_text)
    print('Fix 2 applied: added memory cleanup')
else:
    print('Fix 2 FAILED')

with open('D:/projects/wukong_ai/pathfinding/behavior_clone_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('File updated successfully')
