import h5py
import numpy as np
import os

data_dirs = [
    ('pathfinding_data_balanced', 'v5.5 Training (Balanced)'),
    ('pathfinding_data_noidle', 'v5.4/v5.5 No-Idle'),
    ('pathfinding_data', 'Original Data'),
]

for dname, label in data_dirs:
    dpath = f'D:/projects/wukong_ai/{dname}'
    if not os.path.exists(dpath):
        print(f'\n=== {label}: NOT FOUND ===')
        continue
    
    total_frames = 0
    action_counts = {}
    bucket_counts = {}
    goal_counts = {}
    
    for f in sorted(os.listdir(dpath)):
        if not f.endswith('.h5'):
            continue
        fp = os.path.join(dpath, f)
        try:
            with h5py.File(fp, 'r') as h:
                n = len(h['actions'])
                total_frames += n
                
                acts = h['actions'][:]
                for a in acts:
                    a = int(a)
                    action_counts[a] = action_counts.get(a, 0) + 1
                
                if 'mouse_bucket' in h:
                    buckets = h['mouse_bucket'][:]
                    for b in buckets:
                        b = int(b)
                        bucket_counts[b] = bucket_counts.get(b, 0) + 1
                
                if 'goal_ids' in h:
                    gids = h['goal_ids'][:]
                    for g in gids:
                        g = int(g)
                        goal_counts[g] = goal_counts.get(g, 0) + 1
        except Exception as e:
            print(f'  Error reading {f}: {e}')
    
    print(f'\n=== {label} ({dname}) ===')
    print(f'Total frames: {total_frames}')
    print(f'Files: {len([f for f in os.listdir(dpath) if f.endswith(".h5")])}')
    
    if action_counts:
        print(f'\nAction distribution:')
        action_names = {0: 'forward', 1: 'turn_left', 2: 'turn_right', 3: 'idle', 4: 'dodge'}
        for a in sorted(action_counts.keys()):
            name = action_names.get(a, f'action_{a}')
            pct = action_counts[a] / total_frames * 100
            print(f'  {a} ({name}): {action_counts[a]} ({pct:.1f}%)')
    
    if bucket_counts:
        print(f'\nMouse Bucket distribution:')
        bucket_dxs = {0:-400, 1:-150, 2:-60, 3:0, 4:60, 5:150, 6:400}
        for b in sorted(bucket_counts.keys()):
            dx = bucket_dxs.get(b, '?')
            pct = bucket_counts[b] / total_frames * 100
            print(f'  Bucket {b} (dx={dx}): {bucket_counts[b]} ({pct:.1f}%)')
    
    if goal_counts:
        print(f'\nGoal distribution:')
        for g in sorted(goal_counts.keys()):
            pct = goal_counts[g] / total_frames * 100
            print(f'  Goal {g}: {goal_counts[g]} ({pct:.1f}%)')

# Training log
log_path = 'D:/projects/wukong_ai/training/training_log_v55_optimized_50ep.csv'
if os.path.exists(log_path):
    import csv
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'\n=== v5.5 Training Log (50 epochs) ===')
    print(f'Total rows: {len(rows)}')
    if rows:
        print(f'Columns: {list(rows[0].keys())}')
        for row in rows[:3]:
            print(f"  Epoch {row.get('epoch','?')}: Loss={row.get('loss','?')}, Acc_A={row.get('acc_a','?')}, Acc_M={row.get('acc_m','?')}")
        print('  ...')
        for row in rows[-3:]:
            print(f"  Epoch {row.get('epoch','?')}: Loss={row.get('loss','?')}, Acc_A={row.get('acc_a','?')}, Acc_M={row.get('acc_m','?')}")

# Check checkpoint files
ckpt_dir = 'D:/projects/wukong_ai/checkpoints'
if os.path.exists(ckpt_dir):
    print(f'\n=== Checkpoints ===')
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith('.pt'):
            fp = os.path.join(ckpt_dir, f)
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f'  {f}: {size_mb:.1f} MB')
