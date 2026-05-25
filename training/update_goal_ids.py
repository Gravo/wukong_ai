#!/usr/bin/env python3
"""
更新已有 goal_ids 的文件，根据 goals 属性重新设置 goal_id
映射规则：
  - goals=t1 → goal_id=1
  - goals=t2 → goal_id=2
  - goals=t3 → goal_id=3
  - goals=t4 → goal_id=4
  - goals=t5 → goal_id=5
"""

import h5py
import os
import glob
import numpy as np
import re

DATA_DIR = r'D:\projects\wukong_ai\pathfinding_data'

def update_goal_ids():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.h5')))
    
    print(f'=== Updating Goal IDs ({len(files)} files) ===\n')
    
    updated = 0
    skipped = 0
    errors = 0
    
    for fpath in files:
        basename = os.path.basename(fpath)
        
        try:
            # ============================================
            # 步骤1：读取所有数据和属性
            # ============================================
            with h5py.File(fpath, 'r') as f:
                # 检查必要的数据集
                if 'goal_ids' not in f:
                    print(f'⏭️  {basename}: 无 goal_ids，跳过')
                    skipped += 1
                    continue
                
                if 'goals' not in f.attrs:
                    print(f'⏭️  {basename}: 无 goals 属性，跳过')
                    skipped += 1
                    continue
                
                # 读取 goals 属性
                goals_value = f.attrs['goals']
                if isinstance(goals_value, bytes):
                    goals_str = goals_value.decode('utf-8')
                else:
                    goals_str = str(goals_value)
                
                # 映射 goals → goal_id
                goal_id = None
                if goals_str == 't1':
                    goal_id = 1
                elif goals_str == 't2':
                    goal_id = 2
                elif goals_str == 't3':
                    goal_id = 3
                elif goals_str == 't4':
                    goal_id = 4
                elif goals_str == 't5':
                    goal_id = 5
                else:
                    # 尝试提取数字
                    match = re.search(r'(\d+)', goals_str)
                    if match:
                        goal_id = int(match.group(1))
                    else:
                        print(f'⚠️  {basename}: 无法解析 goals="{goals_str}"，跳过')
                        skipped += 1
                        continue
                
                # 检查是否已经是正确的值
                goal_ids = f['goal_ids'][:]
                unique = set(goal_ids)
                
                if len(unique) == 1 and 0 in unique and goal_id != 0:
                    print(f'🔧 {basename}: goals={goals_str} → 更新 goal_ids {unique} → {goal_id}')
                    
                    # 读取所有数据集
                    frames = f['frames'][:]
                    actions = f['actions'][:]
                    timestamps = f['timestamps'][:]
                    mouse_dx = f['mouse_dx'][:]
                    mouse_dy = f['mouse_dy'][:]
                    mouse_buttons = f['mouse_buttons'][:]
                    
                    # 创建新的 goal_ids
                    num_frames = len(frames)
                    new_goal_ids = np.full(num_frames, goal_id, dtype=np.uint8)
                    
                    # ⚠️ 重要：先保存所有属性到字典！
                    attrs_dict = {}
                    for attr_name in f.attrs.keys():
                        attrs_dict[attr_name] = f.attrs[attr_name]
                else:
                    print(f'✅ {basename}: goal_ids 已经正确 ({unique})，跳过')
                    skipped += 1
                    continue
            
            # ============================================
            # 步骤2：写入临时文件（在 with 块外面）
            # ============================================
            temp_path = fpath + '.tmp'
            with h5py.File(temp_path, 'w') as fout:
                fout.create_dataset('frames', data=frames)
                fout.create_dataset('actions', data=actions)
                fout.create_dataset('timestamps', data=timestamps)
                fout.create_dataset('mouse_dx', data=mouse_dx)
                fout.create_dataset('mouse_dy', data=mouse_dy)
                fout.create_dataset('mouse_buttons', data=mouse_buttons)
                fout.create_dataset('goal_ids', data=new_goal_ids)
                
                # 写入保存的属性
                for attr_name, attr_value in attrs_dict.items():
                    fout.attrs[attr_name] = attr_value
            
            # ============================================
            # 步骤3：替换原文件
            # ============================================
            os.remove(fpath)
            os.rename(temp_path, fpath)
            updated += 1
        
        except Exception as e:
            print(f'❌ {basename}: Error - {e}')
            errors += 1
    
    print(f'\n=== Summary ===')
    print(f'Total files: {len(files)}')
    print(f'Updated: {updated}')
    print(f'Skipped: {skipped}')
    print(f'Errors: {errors}')

if __name__ == '__main__':
    update_goal_ids()
