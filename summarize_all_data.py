import h5py, numpy as np, os

data_dir = 'pathfinding_data'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])

action_names = ['idle', 'attack', 'heavy', 'dodge', 'forward', 'right',
                'left', 'dodge_atk', 'lock', 'heal']

print('=' * 80)
print(f'TOTAL FILES: {len(files)}')
print('=' * 80)
print()

goal_stats = {}
total_frames = 0
qualified_frames = 0

for i, fname in enumerate(files):
    file_path = os.path.join(data_dir, fname)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    
    with h5py.File(file_path, 'r') as f:
        frames = f['frames'][:]
        actions = f['actions'][:]
        goal_ids = f['goal_ids'][:]
        mouse_dx = f['mouse_dx'][:]
        mouse_dy = f['mouse_dy'][:]
        
        n_frames = len(frames)
        total_frames += n_frames
        
        # 统计 Goal IDs
        unique_goals, counts = np.unique(goal_ids, return_counts=True)
        goal_dict = dict(zip(unique_goals, counts))
        
        # 动作分布
        unique_acts, act_counts = np.unique(actions, return_counts=True)
        act_dict = dict(zip(unique_acts, act_counts))
        
        # 鼠标数据质量
        dx_std = mouse_dx.std()
        dy_std = mouse_dy.std()
        mouse_ok = dx_std > 0.01 and dy_std > 0.01
        
        if mouse_ok:
            qualified_frames += n_frames
        
        # 打印每个文件的信息
        print(f'[{i+1}/{len(files)}] {fname}')
        print(f'  Size: {size_mb:.1f} MB | Frames: {n_frames}')
        print(f'  Goal IDs: {dict([(int(g), int(c)) for g, c in zip(unique_goals, counts)])}')
        print(f'  Actions: {", ".join([f"{action_names[int(a)]}:{int(c)}" for a, c in zip(unique_acts, act_counts)])}')
        print(f'  Mouse: dx_std={dx_std:.4f}, dy_std={dy_std:.4f} {"✅" if mouse_ok else "❌"}')
        print()
        
        # 按 Goal ID 统计
        for g, c in zip(unique_goals, counts):
            g = int(g)
            if g not in goal_stats:
                goal_stats[g] = {'files': 0, 'frames': 0, 'mouse_ok': 0}
            goal_stats[g]['files'] += 1
            goal_stats[g]['frames'] += c
            if mouse_ok:
                goal_stats[g]['mouse_ok'] += c
    
    f.close()

# 总结
print('=' * 80)
print('SUMMARY BY GOAL ID')
print('=' * 80)
print()

for g in sorted(goal_stats.keys()):
    stats = goal_stats[g]
    print(f'Goal {g}:')
    print(f'  Files: {stats["files"]}')
    print(f'  Total Frames: {stats["frames"]}')
    print(f'  Qualified Frames (mouse OK): {stats["mouse_ok"]}')
    print()

print('=' * 80)
print('OVERALL STATS')
print('=' * 80)
print(f'Total Files: {len(files)}')
print(f'Total Frames: {total_frames}')
print(f'Qualified Frames (mouse OK): {qualified_frames}')
print(f'Qualification Rate: {qualified_frames / total_frames * 100:.1f}%')
print()

# 检查有多少文件 Goal ID 是 0（未标注）
unlabeled = sum([1 for f in files if 'goal_0' in f or check_goal_id(os.path.join(data_dir, f)) == 0])
print(f'⚠️  Unlabeled files (Goal ID = 0): {unlabeled}')
print()

def check_goal_id(file_path):
    with h5py.File(file_path, 'r') as f:
        return int(np.unique(f['goal_ids'][:])[0])

print('=' * 80)
print('RECOMMENDATIONS')
print('=' * 80)

if 0 in goal_stats:
    print('❌ CRITICAL: Some files have Goal ID = 0 (not labeled!)')
    print('   → DELETE or re-label these files!')
    print()

if qualified_frames / total_frames < 0.8:
    print('⚠️  WARNING: Less than 80% frames have qualified mouse data!')
    print('   → Check files with mouse_dx std < 0.01')
    print()

print('✅ Next steps:')
print('  1. Record more Goal 2 data (target: 2000-3000 frames)')
print('  2. Ensure ALL files have correct Goal ID!')
print('  3. Verify mouse data quality after each recording!')
print()
print('=' * 80)
