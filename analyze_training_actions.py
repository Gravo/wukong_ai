"""统计训练脚本实际使用的动作分布（从 mouse_dx 重新计算）"""
import h5py
import numpy as np
import os

dirs = [
    ('pathfinding_data_balanced', 'v5.5 Balanced (9790 frames)'),
    ('pathfinding_data_noidle', 'v5.4 No-Idle (5582 frames)'),
]

BUCKET_BOUNDS = [-200, -100, -20, 20, 100, 200]
BUCKET_DXS = {0:-400, 1:-150, 2:-60, 3:0, 4:60, 5:150, 6:400}

for dname, label in dirs:
    dpath = f'D:/projects/wukong_ai/{dname}'
    action_counts = {0: 0, 1: 0, 2: 0}  # forward, turn_left, turn_right
    bucket_counts = {i: 0 for i in range(7)}
    total = 0
    mouse_dx_all = []
    
    for f in sorted(os.listdir(dpath)):
        if not f.endswith('.h5'):
            continue
        fp = os.path.join(dpath, f)
        try:
            with h5py.File(fp, 'r') as h:
                dx = h['mouse_dx'][:]
                mouse_dx_all.append(dx)
                
                for x in dx:
                    total += 1
                    if x < -20:
                        action_counts[1] += 1
                    elif x > 20:
                        action_counts[2] += 1
                    else:
                        action_counts[0] += 1
                    
                    # bucket
                    if x <= -200: bucket_counts[0] += 1
                    elif x <= -100: bucket_counts[1] += 1
                    elif x <= -20: bucket_counts[2] += 1
                    elif x <= 20: bucket_counts[3] += 1
                    elif x <= 100: bucket_counts[4] += 1
                    elif x <= 200: bucket_counts[5] += 1
                    else: bucket_counts[6] += 1
        except Exception as e:
            print(f'  Error: {f}: {e}')
    
    mouse_dx_all = np.concatenate(mouse_dx_all)
    
    print(f'\n{"="*60}')
    print(f'{label}')
    print(f'{"="*60}')
    print(f'Total frames: {total}')
    
    print(f'\n--- Action Distribution (from mouse_dx) ---')
    action_names = {0: 'forward', 1: 'turn_left', 2: 'turn_right'}
    for a in sorted(action_counts.keys()):
        pct = action_counts[a] / total * 100
        bar = '#' * int(pct / 2)
        print(f'  {a} ({action_names[a]:>10s}): {action_counts[a]:>5d} ({pct:5.1f}%) {bar}')
    
    print(f'\n--- Mouse Bucket Distribution ---')
    for b in sorted(bucket_counts.keys()):
        pct = bucket_counts[b] / total * 100
        bar = '#' * int(pct / 2)
        print(f'  Bucket {b} (dx={BUCKET_DXS[b]:>4d}): {bucket_counts[b]:>5d} ({pct:5.1f}%) {bar}')
    
    print(f'\n--- Mouse DX Statistics ---')
    print(f'  Mean: {mouse_dx_all.mean():.3f}')
    print(f'  Std:  {mouse_dx_all.std():.3f}')
    print(f'  Min:  {mouse_dx_all.min():.3f}')
    print(f'  Max:  {mouse_dx_all.max():.3f}')
    print(f'  Median: {np.median(mouse_dx_all):.3f}')
    nonzero = mouse_dx_all[mouse_dx_all != 0]
    print(f'  Non-zero count: {len(nonzero)} ({len(nonzero)/total*100:.1f}%)')
    if len(nonzero) > 0:
        print(f'  Non-zero mean: {nonzero.mean():.3f}')
        print(f'  Non-zero std:  {nonzero.std():.3f}')

# Also check original data action encoding
print(f'\n{"="*60}')
print(f'Original Data - Raw Action Encoding')
print(f'{"="*60}')
dpath = 'D:/projects/wukong_ai/pathfinding_data'
action_counts_raw = {}
total_raw = 0
for f in sorted(os.listdir(dpath)):
    if not f.endswith('.h5'):
        continue
    fp = os.path.join(dpath, f)
    with h5py.File(fp, 'r') as h:
        acts = h['actions'][:]
        for a in acts:
            a = int(a)
            action_counts_raw[a] = action_counts_raw.get(a, 0) + 1
            total_raw += 1

action_names_raw = {0: 'forward (W)', 4: 'forward_v2 (W)', 5: 'right (D)', 6: 'left (A)', 8: 'unknown'}
for a in sorted(action_counts_raw.keys()):
    name = action_names_raw.get(a, f'action_{a}')
    pct = action_counts_raw[a] / total_raw * 100
    print(f'  Raw action {a} ({name}): {action_counts_raw[a]} ({pct:.1f}%)')
