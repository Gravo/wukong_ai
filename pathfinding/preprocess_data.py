"""
preprocess_data.py - 预处理寻路数据
将h5原始数据转换为预堆叠npz格式，供behavior_clone.py快速加载
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRAME_STACK


def preprocess(data_dir):
    import h5py
    
    out_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)
    stacked_path = os.path.join(out_dir, "stacked_data.npz")
    
    if os.path.exists(stacked_path):
        print(f"[预处理] {stacked_path} 已存在，跳过", flush=True)
        return stacked_path
    
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[预处理] 未找到h5文件: {data_dir}", flush=True)
        return None
    
    print(f"[预处理] 找到 {len(h5_files)} 个h5文件", flush=True)
    
    all_stacked = []
    all_actions = []
    all_mouse = []
    
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            frames = f["frames"][:]
            actions = f["actions"][:]
            has_mouse = f.attrs.get("has_mouse", False) and "mouse_dx" in f
            if has_mouse:
                mouse_dx = f["mouse_dx"][:]
                mouse_dy = f["mouse_dy"][:]
                mouse_btns = f["mouse_buttons"][:]
            else:
                mouse_dx = np.zeros(len(frames), dtype=np.float32)
                mouse_dy = np.zeros(len(frames), dtype=np.float32)
                mouse_btns = np.zeros(len(frames), dtype=np.uint8)
            
            n = len(frames)
            print(f"  {os.path.basename(h5_path)}: {n} frames, mouse={has_mouse}", flush=True)
            
            for i in range(FRAME_STACK - 1, n):
                stacked = np.concatenate(
                    [frames[j].transpose(2, 0, 1) for j in range(i - FRAME_STACK + 1, i + 1)],
                    axis=0,
                )
                all_stacked.append(stacked)
                all_actions.append(int(actions[i]))
                all_mouse.append([
                    np.clip(float(mouse_dx[i]) / 100.0, -1.0, 1.0),
                    np.clip(float(mouse_dy[i]) / 100.0, -1.0, 1.0),
                    float((int(mouse_btns[i]) & 1) > 0),
                    float((int(mouse_btns[i]) & 2) > 0),
                ])
    
    all_stacked = np.array(all_stacked, dtype=np.uint8)
    all_actions = np.array(all_actions, dtype=np.int64)
    all_mouse = np.array(all_mouse, dtype=np.float32)
    
    print(f"[预处理] 堆叠帧: {all_stacked.shape} ({all_stacked.nbytes/1024/1024:.1f} MB)", flush=True)
    
    np.savez_compressed(stacked_path, frames=all_stacked, actions=all_actions, mouse=all_mouse)
    sz = os.path.getsize(stacked_path)
    print(f"[预处理] 保存: {stacked_path} ({sz/1024/1024:.1f} MB)", flush=True)
    
    # 释放内存
    del all_stacked
    return stacked_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="pathfinding_data")
    args = parser.parse_args()
    preprocess(args.data_dir)
