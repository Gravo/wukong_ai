import h5py, numpy as np, os

data_dir = 'pathfinding_data'

# 获取最新的 2 个文件
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')], 
                key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), 
                reverse=True)[:2]

print('=' * 80)
print(f'Checking latest {len(files)} files...')
print('=' * 80)
print()

action_names = ['idle', 'attack', 'heavy', 'dodge', 'forward', 'right',
                'left', 'dodge_atk', 'lock', 'heal']

modified_files = []

for i, fname in enumerate(files):
    file_path = os.path.join(data_dir, fname)
    size_mb = os.path.getsize(file_path) / 1024 / 1024
    
    print(f'[{i+1}/{len(files)}] {fname}')
    print(f'  Size: {size_mb:.1f} MB')
    
    with h5py.File(file_path, 'r') as f:
        frames = f['frames'][:]
        actions = f['actions'][:]
        goal_ids = f['goal_ids'][:]
        mouse_dx = f['mouse_dx'][:]
        mouse_dy = f['mouse_dy'][:]
        
        n_frames = len(frames)
        print(f'  Frames: {n_frames}')
        
        # 检查 Goal IDs
        unique_goals, counts = np.unique(goal_ids, return_counts=True)
        goal_dict = dict(zip(unique_goals, counts))
        print(f'  Goal IDs: {dict([(int(g), int(c)) for g, c in zip(unique_goals, counts)])}')
        
        # 检查是否需要修改
        if 0 in unique_goals and len(unique_goals) == 1:
            print(f'  ❌ Goal ID = 0 (not set!) Will modify to 2...')
            
            # 修改 Goal ID
            f.close()  # 先关闭读模式
            with h5py.File(file_path, 'r+') as f_write:  # 重新以读写模式打开
                f_write['goal_ids'][:] = 2
                new_goal_ids = f_write['goal_ids'][:]
                if new_goal_ids.min() == 2 and new_goal_ids.max() == 2:
                    print(f'  ✅ SUCCESS: Modified Goal ID to 2!')
                    modified_files.append(fname)
                else:
                    print(f'  ❌ ERROR: Failed to modify Goal ID!')
            
            # 重新打开以继续检查
            with h5py.File(file_path, 'r') as f:
                goal_ids = f['goal_ids'][:]
                unique_goals, counts = np.unique(goal_ids, return_counts=True)
        else:
            print(f'  ✅ Goal ID already set (not 0)')
        
        # 动作分布
        unique_acts, act_counts = np.unique(actions, return_counts=True)
        act_dict = dict(zip(unique_acts, act_counts))
        print(f'  Actions: {", ".join([f"{action_names[int(a)]}:{int(c)}" for a, c in zip(unique_acts, act_counts)])}')
        
        # 鼠标数据质量
        dx_std = mouse_dx.std()
        dy_std = mouse_dy.std()
        mouse_ok = dx_std > 0.01 and dy_std > 0.01
        
        print(f'  Mouse: dx_std={dx_std:.4f}, dy_std={dy_std:.4f} {"✅" if mouse_ok else "❌"}')
        
        if not mouse_ok:
            print(f'  ⚠️  WARNING: Mouse data missing or very low!')
        
        print()

print('=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Modified {len(modified_files)} file(s) to Goal ID = 2:')
for f in modified_files:
    print(f'  - {f}')
print()
print('Done!')
print('=' * 80)
