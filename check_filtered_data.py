"""
check_filtered_data.py - 检查过滤后数据的分布
"""

import h5py
import numpy as np
from pathlib import Path
from collections import Counter

INPUT_DIR = "D:/projects/wukong_ai/pathfinding_data_noidle"

ACTION_NAMES = {
    0: "idle",
    3: "dodge",
    4: "forward",
    5: "turn_left",
    6: "turn_right",
}

def main():
    input_dir = Path(INPUT_DIR)
    h5_files = sorted(input_dir.glob("*.h5"))
    
    print(f"="*60, flush=True)
    print(f"检查过滤后数据分布", flush=True)
    print(f"="*60, flush=True)
    print(f"目录: {INPUT_DIR}", flush=True)
    print(f"找到 {len(h5_files)} 个 .h5 文件", flush=True)
    print(flush=True)
    
    total_frames = 0
    action_counter = Counter()
    mouse_dx_list = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            actions = f["actions"][:]
            frames = f["frames"][:]
            
            total_frames += len(actions)
            action_counter.update(actions.tolist())
            
            # 读取 mouse_dx (如果存在)
            if "mouse_dx" in f:
                mouse_dx = f["mouse_dx"][:]
                mouse_dx_list.extend(mouse_dx.tolist())
        
        print(f"  {h5_file.name}: {len(actions)} 帧", flush=True)
    
    # 打印统计
    print(flush=True)
    print(f"="*60, flush=True)
    print(f"数据分布", flush=True)
    print(f"="*60, flush=True)
    print(f"总帧数: {total_frames}", flush=True)
    print(flush=True)
    
    for action_id in sorted(action_counter.keys()):
        count = action_counter[action_id]
        name = ACTION_NAMES.get(action_id, f"unknow_{action_id}")
        pct = count / total_frames * 100
        print(f"  {name:15s} ({action_id:2d}): {count:6d} 帧 ({pct:5.1f}%)", flush=True)
    
    # mouse_dx 统计
    if len(mouse_dx_list) > 0:
        mouse_dx_arr = np.array(mouse_dx_list)
        print(flush=True)
        print(f"="*60, flush=True)
        print(f"Mouse DX 统计", flush=True)
        print(f"="*60, flush=True)
        print(f"  数量: {len(mouse_dx_arr)}", flush=True)
        print(f"  均值: {mouse_dx_arr.mean():.1f}", flush=True)
        print(f"  标准差: {mouse_dx_arr.std():.1f}", flush=True)
        print(f"  最小值: {mouse_dx_arr.min():.1f}", flush=True)
        print(f"  25%: {np.percentile(mouse_dx_arr, 25):.1f}", flush=True)
        print(f"  中位数: {np.median(mouse_dx_arr):.1f}", flush=True)
        print(f"  75%: {np.percentile(mouse_dx_arr, 75):.1f}", flush=True)
        print(f"  最大值: {mouse_dx_arr.max():.1f}", flush=True)
        
        # 左右分布
        left = (mouse_dx_arr < -50).sum()
        center = ((mouse_dx_arr >= -50) & (mouse_dx_arr <= 50)).sum()
        right = (mouse_dx_arr > 50).sum()
        
        print(flush=True)
        print(f"  左转 (< -50): {left} 帧 ({left/len(mouse_dx_arr)*100:.1f}%)", flush=True)
        print(f"  中间 (-50~50): {center} 帧 ({center/len(mouse_dx_arr)*100:.1f}%)", flush=True)
        print(f"  右转 (> 50): {right} 帧 ({right/len(mouse_dx_arr)*100:.1f}%)", flush=True)
    
    print(flush=True)
    print(f"="*60, flush=True)


if __name__ == "__main__":
    main()
