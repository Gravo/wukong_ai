"""
preprocess_data.py - 预处理寻路数据 v3
逐个h5文件处理，每个文件输出一个npz分片（帧堆叠）。
每个分片单独加载到内存，不会同时把所有数据加载。

用法：预处理完成后直接运行 behavior_clone_v2.py
"""
import os
import sys
import glob
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRAME_STACK


def preprocess(data_dir):
    import h5py

    out_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if not h5_files:
        print(f"[预处理] 未找到h5文件: {data_dir}", flush=True)
        return None

    print(f"[预处理] 找到 {len(h5_files)} 个h5文件", flush=True)

    index = []
    total_samples = 0
    total_actions = np.zeros(10, dtype=np.int64)

    for fi, h5_path in enumerate(h5_files):
        print(f"  [{fi+1}/{len(h5_files)}] {os.path.basename(h5_path)}...", end=" ", flush=True)

        with h5py.File(h5_path, "r") as f:
            frames = f["frames"][:]       # (N, 224, 224, 3) uint8
            actions = f["actions"][:]     # (N,) int8
            n = len(frames)

            has_mouse = f.attrs.get("has_mouse", False) and "mouse_dx" in f
            if has_mouse:
                mouse_dx = f["mouse_dx"][:]
                mouse_dy = f["mouse_dy"][:]
                mouse_btns = f["mouse_buttons"][:]
            else:
                mouse_dx = np.zeros(n, dtype=np.float32)
                mouse_dy = np.zeros(n, dtype=np.float32)
                mouse_btns = np.zeros(n, dtype=np.uint8)

        # 帧堆叠：一次只处理一个文件，内存可控
        n_samples = max(0, n - FRAME_STACK + 1)
        stacked = np.zeros((n_samples, FRAME_STACK * 3, 224, 224), dtype=np.uint8)
        act_arr = np.zeros(n_samples, dtype=np.int64)
        mouse_arr = np.zeros((n_samples, 4), dtype=np.float32)

        for i in range(n_samples):
            # 堆叠4帧
            frame_idx = i + FRAME_STACK - 1
            parts = []
            for j in range(FRAME_STACK):
                parts.append(frames[frame_idx - FRAME_STACK + 1 + j].transpose(2, 0, 1))
            stacked[i] = np.concatenate(parts, axis=0)
            act_arr[i] = int(actions[frame_idx])
            mouse_arr[i] = [
                np.clip(float(mouse_dx[frame_idx]) / 100.0, -1.0, 1.0),
                np.clip(float(mouse_dy[frame_idx]) / 100.0, -1.0, 1.0),
                float((int(mouse_btns[frame_idx]) & 1) > 0),
                float((int(mouse_btns[frame_idx]) & 2) > 0),
            ]

        # 保存为npz分片
        shard_path = os.path.join(out_dir, f"shard_{fi:03d}.npz")
        np.savez_compressed(shard_path, frames=stacked, actions=act_arr, mouse=mouse_arr)
        sz = os.path.getsize(shard_path)

        counts = np.bincount(act_arr, minlength=10)
        total_actions += counts
        total_samples += n_samples

        index.append({
            "shard": f"shard_{fi:03d}.npz",
            "samples": int(n_samples),
        })

        # 释放内存
        del frames, actions, stacked, act_arr, mouse_arr
        del mouse_dx, mouse_dy, mouse_btns

        print(f"{n_samples} samples, {sz/1024/1024:.1f}MB", flush=True)

    # 保存索引
    with open(os.path.join(out_dir, "shard_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n[预处理] 完成", flush=True)
    print(f"  总样本: {total_samples}, 分片: {len(h5_files)}", flush=True)
    print(f"  动作分布:", flush=True)
    names = ['idle','atk','heavy','dodge','fwd','rgt','lft','d_atk','lock','heal']
    for i in range(10):
        if total_actions[i] > 0:
            pct = total_actions[i] / total_actions.sum() * 100
            print(f"    {names[i]:8s}: {total_actions[i]} ({pct:.1f}%)", flush=True)

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="pathfinding_data")
    args = parser.parse_args()
    preprocess(args.data_dir)
