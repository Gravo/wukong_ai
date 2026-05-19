"""分步内存测试"""
import os, sys, gc, psutil
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRAME_STACK, BC

def mem():
    return psutil.Process().memory_info().rss / 1024**2

# 找最大h5
h5_files = sorted(glob.glob(os.path.join(BC["data_dir"], "*.h5"))) if 'glob' in dir() else []
import glob as g
h5_files = sorted(g.glob(os.path.join(BC["data_dir"], "*.h5")))
h5_path = h5_files[5]

print(f"1. 测试前: {mem():.0f}MB")

with h5py.File(h5_path, "r") as f:
    print(f"2. 打开h5后: {mem():.0f}MB")
    frames = f["frames"][:]
    print(f"3. 读入frames后: {mem():.0f}MB (shape={frames.shape}, dtype={frames.dtype})")
    actions = f["actions"][:]
    mdx = f["mouse_dx"][:]
    mdy = f["mouse_dy"][:]
    print(f"4. 读入全部后: {mem():.0f}MB")

del actions, mdx, mdy
gc.collect()
print(f"5. 删除部分后: {mem():.0f}MB")

# 只保留frames
print(f"\n6. frames占内存: {frames.nbytes/1024**2:.0f}MB")
print(f"7. 剩余: {mem():.0f}MB")
