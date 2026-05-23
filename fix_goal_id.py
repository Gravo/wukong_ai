import h5py
import os

data_dir = 'pathfinding_data'
file_name = 'pathfinding_ep1_1779541581.h5'
file_path = os.path.join(data_dir, file_name)

print(f'Processing: {file_name}')
print(f'File size: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB')

# 打开文件（读+写模式）
with h5py.File(file_path, 'r+') as f:
    # 检查 goal_ids 数据集是否存在
    if 'goal_ids' not in f:
        print('❌ ERROR: goal_ids dataset not found!')
        exit(1)
    
    # 读取当前的 goal_ids
    old_goal_ids = f['goal_ids'][:]
    print(f'Old goal_ids: min={old_goal_ids.min()}, max={old_goal_ids.max()}, unique={list(set(old_goal_ids))}')
    
    # 修改 goal_ids 为 2
    f['goal_ids'][:] = 2
    
    # 验证修改是否成功
    new_goal_ids = f['goal_ids'][:]
    print(f'New goal_ids: min={new_goal_ids.min()}, max={new_goal_ids.max()}, unique={list(set(new_goal_ids))}')
    
    if new_goal_ids.min() == 2 and new_goal_ids.max() == 2:
        print('✅ SUCCESS: All goal_ids set to 2!')
    else:
        print('❌ ERROR: goal_ids not correctly modified!')

print()
print('=' * 60)
print('Done!')
print('=' * 60)
