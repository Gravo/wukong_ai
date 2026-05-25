import h5py, numpy as np, os

data_dir = 'pathfinding_data'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')],
               key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)

# 取最新的 2 个文件
latest_2 = files[:2]

action_names = ['idle', 'attack', 'heavy', 'dodge', 'forward', 'right',
                'left', 'dodge_atk', 'lock', 'heal']

for i, fname in enumerate(latest_2):
    file_path = os.path.join(data_dir, fname)
    print('=' * 70)
    print(f'File {i+1}: {fname}')
    print(f'Size: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB')
    print('=' * 70)
    
    with h5py.File(file_path, 'r') as f:
        frames = f['frames'][:]
        actions = f['actions'][:]
        goal_ids = f['goal_ids'][:]
        mouse_dx = f['mouse_dx'][:]
        mouse_dy = f['mouse_dy'][:]
        
        print(f'Total Frames: {len(frames)}')
        print()
        print('--- Goal IDs ---')
        unique_goals, counts = np.unique(goal_ids, return_counts=True)
        for g, c in zip(unique_goals, counts):
            pct = c / len(goal_ids) * 100
            print(f'  Goal {int(g)}: {c} frames ({pct:.1f}%)')
        
        # 检查是否包含 Goal 2
        if 2 in unique_goals:
            print('  ✅ Goal 2 detected!')
        else:
            print('  ❌ Goal 2 NOT detected! (Was G key pressed?)')
        
        print()
        print('--- Action Distribution ---')
        unique_acts, act_counts = np.unique(actions, return_counts=True)
        for a, c in zip(unique_acts, act_counts):
            name = action_names[int(a)] if int(a) < len(action_names) else f'unknown_{a}'
            pct = c / len(actions) * 100
            print(f'  {name}: {c} ({pct:.1f}%)')
        print()
        print('--- Mouse Data ---')
        print(f'  dx: mean={mouse_dx.mean():.4f}, std={mouse_dx.std():.4f}')
        print(f'      min={mouse_dx.min():.4f}, max={mouse_dx.max():.4f}')
        print(f'  dy: mean={mouse_dy.mean():.4f}, std={mouse_dy.std():.4f}')
        print(f'      min={mouse_dy.min():.4f}, max={mouse_dy.max():.4f}')
        
        if mouse_dx.std() < 0.001 and mouse_dy.std() < 0.001:
            print('  ❌ WARNING: Mouse data is nearly ZERO! Re-record!')
        elif mouse_dx.std() < 0.01:
            print('  ⚠️  CAUTION: Mouse data very low')
        else:
            print('  ✅ Mouse data looks OK')
        
        print()
        print('--- First 20 frames mouse ---')
        print(f'  dx: {mouse_dx[:20].tolist()}')
        print(f'  dy: {mouse_dy[:20].tolist()}')
        print()
        print()
