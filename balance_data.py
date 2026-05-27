"""
balance_data.py - 平衡训练数据分布

解决数据失衡问题（idle+forward 占 87.6%）

策略：
1. 过滤 idle 帧（保留部分）
2. 过采样转向帧
3. 欠采样 forward 帧
4. 输出平衡后的数据集

用法：
    python balance_data.py --input-dir pathfinding_data --output-dir pathfinding_data_balanced
"""
import os
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import Counter


def analyze_data(input_dir):
    """分析数据分布"""
    input_path = Path(input_dir)
    h5_files = list(input_path.glob("*.h5"))

    print(f"\n{'='*60}")
    print(f"  数据分布分析")
    print(f"{'='*60}")
    print(f"  找到 {len(h5_files)} 个 h5 文件\n")

    total_actions = Counter()
    total_frames = 0
    file_stats = []

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            actions = f['actions'][:]
            n = len(actions)
            total_frames += n

            action_counts = Counter(actions)
            total_actions.update(action_counts)

            file_stats.append({
                'file': h5_file.name,
                'frames': n,
                'actions': action_counts,
            })

    # 打印总体统计
    print(f"  总帧数: {total_frames}")
    print(f"\n  动作分布:")

    action_names = {
        0: "idle",
        1: "forward",
        2: "turn_right",
        3: "turn_left",
        4: "dodge",
    }

    for action_id in sorted(total_actions.keys()):
        count = total_actions[action_id]
        pct = count / total_frames * 100
        name = action_names.get(action_id, f"action_{action_id}")
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"    {name:12s}: {count:6d} ({pct:5.1f}%) {bar}")

    return total_actions, total_frames, file_stats


def balance_data(input_dir, output_dir, idle_ratio=0.2, forward_ratio=0.5, oversample_turns=3):
    """
    平衡数据分布

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        idle_ratio: idle 帧保留比例
        forward_ratio: forward 帧保留比例
        oversample_turns: 转向帧过采样倍数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    h5_files = list(input_path.glob("*.h5"))
    print(f"\n{'='*60}")
    print(f"  数据平衡处理")
    print(f"{'='*60}")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"  策略: idle={idle_ratio*100:.0f}%, forward={forward_ratio*100:.0f}%, turns={oversample_turns}x")
    print(f"{'='*60}\n")

    all_frames = []
    all_actions = []
    all_goals = []

    # 读取所有数据
    for h5_file in h5_files:
        print(f"  读取: {h5_file.name}...")
        with h5py.File(h5_file, 'r') as f:
            frames = f['frames'][:]
            actions = f['actions'][:]
            goals = f['goals'][:] if 'goals' in f else None

            all_frames.append(frames)
            all_actions.append(actions)
            if goals is not None:
                all_goals.append(goals)

    # 合并所有数据
    all_frames = np.concatenate(all_frames, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    if all_goals:
        all_goals = np.concatenate(all_goals, axis=0)
    else:
        all_goals = np.zeros(len(all_frames), dtype=np.int64)

    total = len(all_frames)
    print(f"\n  总帧数: {total}")

    # 创建平衡掩码
    balanced_mask = np.ones(total, dtype=bool)

    # 处理 idle 帧
    idle_mask = all_actions == 0
    idle_indices = np.where(idle_mask)[0]
    keep_idle = int(len(idle_indices) * idle_ratio)
    if keep_idle < len(idle_indices):
        np.random.shuffle(idle_indices)
        remove_idle = idle_indices[keep_idle:]
        balanced_mask[remove_idle] = False
        print(f"  idle: {len(idle_indices)} -> {keep_idle} (移除 {len(remove_idle)})")

    # 处理 forward 帧
    forward_mask = all_actions == 1
    forward_indices = np.where(forward_mask)[0]
    keep_forward = int(len(forward_indices) * forward_ratio)
    if keep_forward < len(forward_indices):
        np.random.shuffle(forward_indices)
        remove_forward = forward_indices[keep_forward:]
        balanced_mask[remove_forward] = False
        print(f"  forward: {len(forward_indices)} -> {keep_forward} (移除 {len(remove_forward)})")

    # 应用掩码
    balanced_frames = all_frames[balanced_mask]
    balanced_actions = all_actions[balanced_mask]
    balanced_goals = all_goals[balanced_mask]

    # 过采样转向帧
    turn_mask = (balanced_actions == 2) | (balanced_actions == 3)
    turn_indices = np.where(turn_mask)[0]

    if len(turn_indices) > 0 and oversample_turns > 1:
        # 重复转向帧
        oversample_indices = np.tile(turn_indices, oversample_turns - 1)
        balanced_frames = np.concatenate([balanced_frames, balanced_frames[oversample_indices]], axis=0)
        balanced_actions = np.concatenate([balanced_actions, balanced_actions[oversample_indices]], axis=0)
        balanced_goals = np.concatenate([balanced_goals, balanced_goals[oversample_indices]], axis=0)
        print(f"  turns: {len(turn_indices)} -> {len(turn_indices) * oversample_turns} (过采样 {oversample_turns}x)")

    # 打印平衡后的分布
    print(f"\n  平衡后总帧数: {len(balanced_frames)}")
    print(f"  平衡后分布:")

    action_names = {0: "idle", 1: "forward", 2: "turn_right", 3: "turn_left", 4: "dodge"}
    for action_id in sorted(set(balanced_actions)):
        count = np.sum(balanced_actions == action_id)
        pct = count / len(balanced_actions) * 100
        name = action_names.get(action_id, f"action_{action_id}")
        print(f"    {name:12s}: {count:6d} ({pct:5.1f}%)")

    # 保存平衡后的数据
    output_file = output_path / "balanced_data.h5"
    print(f"\n  保存到: {output_file}")

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('frames', data=balanced_frames, compression='gzip')
        f.create_dataset('actions', data=balanced_actions)
        f.create_dataset('goals', data=balanced_goals)
        f.attrs['total_frames'] = len(balanced_frames)
        f.attrs['source'] = str(input_dir)
        f.attrs['idle_ratio'] = idle_ratio
        f.attrs['forward_ratio'] = forward_ratio
        f.attrs['oversample_turns'] = oversample_turns

    print(f"  完成！")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="平衡训练数据分布")
    parser.add_argument("--input-dir", type=str, default="pathfinding_data",
                        help="输入数据目录")
    parser.add_argument("--output-dir", type=str, default="pathfinding_data_balanced",
                        help="输出数据目录")
    parser.add_argument("--idle-ratio", type=float, default=0.2,
                        help="idle 帧保留比例 (默认 0.2)")
    parser.add_argument("--forward-ratio", type=float, default=0.5,
                        help="forward 帧保留比例 (默认 0.5)")
    parser.add_argument("--oversample-turns", type=int, default=3,
                        help="转向帧过采样倍数 (默认 3)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析数据分布，不处理")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_data(args.input_dir)
    else:
        balance_data(
            args.input_dir,
            args.output_dir,
            args.idle_ratio,
            args.forward_ratio,
            args.oversample_turns,
        )


if __name__ == "__main__":
    main()
