"""
convert_goal_id_to_zero_indexed.py - 将 goal_id 从 1-indexed 转换为 0-indexed
使用：
  C:\Python\python.exe -u convert_goal_id_to_zero_indexed.py --data-dir pathfinding_data
  C:\Python\python.exe -u convert_goal_id_to_zero_indexed.py --dagger-file pathfinding_dagger_round3.h5
"""
import argparse, os, h5py, glob, numpy as np

def convert_human_data(data_dir):
    """转换人类数据（逐帧 goal_ids）"""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    print(f"[Convert] Found {len(h5_files)} h5 files in {data_dir}", flush=True)
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r+') as f:
                if 'goal_ids' in f:
                    goal_ids = f['goal_ids'][:]
                    if goal_ids.min() >= 1:  # 1-indexed
                        goal_ids -= 1  # 转换为 0-indexed
                        del f['goal_ids']
                        f.create_dataset('goal_ids', data=goal_ids)
                        print(f"[Convert] ✅ {os.path.basename(h5_file)}: goal_ids converted to 0-indexed", flush=True)
                    elif goal_ids.min() == 0:
                        print(f"[Convert] ⏭️  {os.path.basename(h5_file)}: already 0-indexed, skipping", flush=True)
                    else:
                        print(f"[Convert] ⚠️  {os.path.basename(h5_file)}: unexpected goal_ids range [{goal_ids.min()}, {goal_ids.max()}], skipping", flush=True)
                else:
                    print(f"[Convert] ⚠️  {os.path.basename(h5_file)}: no goal_ids field, skipping", flush=True)
        except Exception as e:
            print(f"[Convert] ❌ {os.path.basename(h5_file)}: {e}", flush=True)

def convert_dagger_data(dagger_file):
    """转换DAgger数据（文件属性 goal_id）"""
    try:
        with h5py.File(dagger_file, 'r+') as f:
            if 'goal_id' in f.attrs:
                goal_id = f.attrs['goal_id']
                if goal_id >= 1:  # 1-indexed
                    f.attrs['goal_id'] = goal_id - 1  # 转换为 0-indexed
                    print(f"[Convert] ✅ {os.path.basename(dagger_file)}: goal_id converted to 0-indexed ({goal_id} → {goal_id-1})", flush=True)
                elif goal_id == 0:
                    print(f"[Convert] ⏭️  {os.path.basename(dagger_file)}: already 0-indexed, skipping", flush=True)
                else:
                    print(f"[Convert] ⚠️  {os.path.basename(dagger_file)}: unexpected goal_id={goal_id}, skipping", flush=True)
            else:
                print(f"[Convert] ⚠️  {os.path.basename(dagger_file)}: no goal_id attribute, skipping", flush=True)
    except Exception as e:
        print(f"[Convert] ❌ {os.path.basename(dagger_file)}: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="Directory containing human data h5 files")
    parser.add_argument("--dagger-file", type=str, help="DAgger h5 file to convert")
    args = parser.parse_args()
    
    if args.data_dir:
        convert_human_data(args.data_dir)
    
    if args.dagger_file:
        convert_dagger_data(args.dagger_file)
    
    if not args.data_dir and not args.dagger_file:
        print("[Convert] ❌ Please specify --data-dir or --dagger-file", flush=True)
