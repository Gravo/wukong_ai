#!/usr/bin/env python3
"""
修复现有 h5 文件的 goal_ids
- 删除没有 goal_ids 且无法修复的文件
- 根据 goals 属性设置 goal_id:
  - goals=t1 → goal_id=1
  - goals=t2 → goal_id=2
  - goals=t4 → goal_id=4 (或其他映射)
  - 其他 → goal_id=0
"""

import h5py
import os
import glob
import sys
import numpy as np

DATA_DIR = r'D:\projects\wukong_ai\pathfinding_data'

def fix_goal_ids():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))
    
    print(f'=== Fixing Goal IDs ({len(files)} files) ===\n')
    
    fixed = 0
    deleted = 0
    skipped = 0
    
    for fpath in files:
        basename = os.path.basename(fpath)
        
        try:
            with h5py.File(fpath, 'r') as f:
                has_goal_ids = 'goal_ids' in f
                has_goals_attr = 'goals' in f.attrs
                
                # 情况1：已有 goal_ids，跳过
                if has_goal_ids:
                    goal_ids = f['goal_ids'][:]
                    unique = set(goal_ids)
                    print(f'✅ {basename}: goal_ids exists, unique={unique}')
                    skipped += 1
                    continue
                
                # 情况2：没有 goal_ids，但有 goals 属性 → 可以修复
                if has_goals_attr:
                    goals_value = f.attrs['goals']
                    num_frames = f.attrs['num_frames']
                    
                    # 解析 goals 属性，映射到 goal_id
                    # 你的数据：goals=t1, t2, t4, t5 等
                    goal_id = 0  # 默认
                    if isinstance(goals_value, bytes):
                        goals_str = goals_value.decode('utf-8')
                    else:
                        goals_str = str(goals_value)
                    
                    # 映射规则（你可以修改这里）
                    if goals_str == 't1':
                        goal_id = 1
                    elif goals_str == 't2':
                        goal_id = 2
                    elif goals_str == 't4':
                        goal_id = 4
                    elif goals_str == 't5':
                        goal_id = 5
                    else:
                        # 尝试提取数字
                        import re
                        match = re.search(r'(\d+)', goals_str)
                        if match:
                            goal_id = int(match.group(1))
                    
                    print(f'🔧 {basename}: goals={goals_str} → goal_id={goal_id}, frames={num_frames}')
                    
                    # 读取所有数据
                    frames = f['frames'][:]
                    actions = f['actions'][:]
                    timestamps = f['timestamps'][:]
                    mouse_dx = f['mouse_dx'][:]
                    mouse_dy = f['mouse_dy'][:]
                    mouse_buttons = f['mouse_buttons'][:]
                    
                    # 创建 goal_ids
                    goal_ids = np.full(num_frames, goal_id, dtype=np.uint8)
                    
                    # 写入新文件（先写临时文件，再替换）
                    temp_path = fpath + '.tmp'
                    with h5py.File(temp_path, 'w') as fout:
                        fout.create_dataset('frames', data=frames)
                        fout.create_dataset('actions', data=actions)
                        fout.create_dataset('timestamps', data=timestamps)
                        fout.create_dataset('mouse_dx', data=mouse_dx)
                        fout.create_dataset('mouse_dy', data=mouse_dy)
                        fout.create_dataset('mouse_buttons', data=mouse_buttons)
                        fout.create_dataset('goal_ids', data=goal_ids)
                        
                        # 复制属性
                        for attr_name in f.attrs.keys():
                            fout.attrs[attr_name] = f.attrs[attr_name]
                    
                    # 替换原文件
                    os.remove(fpath)
                    os.rename(temp_path, fpath)
                    fixed += 1
                    continue
                
                # 情况3：既没有 goal_ids，也没有 goals 属性 → 无法修复，删除
                print(f'🗑️  {basename}: NO goal_ids, NO goals attr → DELETING')
                os.remove(fpath)
                deleted += 1
        
        except Exception as e:
            print(f'❌ {basename}: Error - {e}')
            deleted += 1  # 有问题的文件也删除
    
    print(f'\n=== Summary ===')
    print(f'Total files: {len(files)}')
    print(f'Fixed (added goal_ids): {fixed}')
    print(f'Skipped (already has goal_ids): {skipped}')
    print(f'Deleted (no goal_ids, no goals attr): {deleted}')

if __name__ == '__main__':
    fix_goal_ids()
