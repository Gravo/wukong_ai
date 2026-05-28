"""oversample_dataset.py - 数据增强:Oversample 非 bucket 3 样本

功能:
  1. 读取 pathfinding_data_noidle 所有 h5 文件
  2. 对非 bucket 3 样本重复采样 5 次
  3. 让 bucket 3 占比从 81% → ~30%
  4. 保存增强后的数据集到新目录

使用方法:
  C:\Python\python.exe -u training\oversample_dataset.py
"""

import h5py
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ============ 配置 ============
INPUT_DIR = r"D:\projects\wukong_ai\pathfinding_data_noidle"
OUTPUT_DIR = r"D:\projects\wukong_ai\pathfinding_data_balanced"
BUCKET_3_ID = 3  # 直行(需要降低占比)
OVERSAMPLE_TIMES = 5  # 非 bucket 3 样本重复采样次数

# ============ Bucket 分类函数 ============
def classify_bucket(mouse_dx):
    """将 mouse_dx 归类到 7 个 bucket"""
    if mouse_dx <= -200:
        return 0
    elif mouse_dx <= -100:
        return 1
    elif mouse_dx <= -20:
        return 2
    elif mouse_dx <= 20:
        return 3
    elif mouse_dx <= 100:
        return 4
    elif mouse_dx <= 200:
        return 5
    else:
        return 6

# ============ 主函数 ============
def main():
    print("=" * 60)
    print("数据增强:Oversample 非 bucket 3 样本")
    print("=" * 60)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    # 创建输出目录
    # 直接覆盖(不询问)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[信息] 已删除旧目录: {OUTPUT_DIR}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有 h5 文件
    h5_files = sorted(input_dir.glob("*.h5"))

    if not h5_files:
        print(f"[错误] 未找到 h5 文件: {INPUT_DIR}")
        return

    print(f"\n[分析] 找到 {len(h5_files)} 个 h5 文件")

    # 统计数据
    total_original = 0
    total_balanced = 0
    bucket_counts_original = [0] * 7
    bucket_counts_balanced = [0] * 7

    # 逐文件处理
    for h5_file in tqdm(h5_files, desc="处理文件"):
        try:
            with h5py.File(h5_file, 'r') as f:
                # 检查必需的 dataset 是否存在
                required_datasets = ['frames', 'actions', 'mouse_dx', 'mouse_dy']
                missing = [ds for ds in required_datasets if ds not in f]
                
                if missing:
                    print(f"\n[警告] 跳过 {h5_file.name} (缺少 dataset: {missing})")
                    continue
                
                # 读取数据（必需）
                frames = f['frames'][:]
                actions = f['actions'][:]
                mouse_dx = f['mouse_dx'][:]
                mouse_dy = f['mouse_dy'][:]
                
                # 读取数据（可选：goals/infos 可能不存在）
                has_goals = 'goals' in f
                has_infos = 'infos' in f
                
                if has_goals:
                    goals = f['goals'][:]
                if has_infos:
                    infos = f['infos'][:]

                n_samples = len(frames)
                total_original += n_samples

                # 分类到 bucket
                bucket_ids = np.array([classify_bucket(dx) for dx in mouse_dx])

                # 统计原始分布
                for bucket_id in bucket_ids:
                    bucket_counts_original[bucket_id] += 1

                # 数据增强:非 bucket 3 样本重复采样
                oversampled_indices = []

                for i in range(n_samples):
                    bucket_id = bucket_ids[i]

                    if bucket_id == BUCKET_3_ID:
                        # bucket 3: 只保留 1 次
                        oversampled_indices.append(i)
                    else:
                        # 非 bucket 3: 重复采样 OVERSAMPLE_TIMES 次
                        for _ in range(OVERSAMPLE_TIMES):
                            oversampled_indices.append(i)

                # 按增强后的索引排序(保持时间顺序)
                oversampled_indices = sorted(oversampled_indices)
                n_balanced = len(oversampled_indices)
                total_balanced += n_balanced

                # 统计增强后分布
                for idx in oversampled_indices:
                    bucket_id = bucket_ids[idx]
                    bucket_counts_balanced[bucket_id] += 1

                # 保存增强后的数据
                output_file = output_dir / h5_file.name

                with h5py.File(output_file, 'w') as f_out:
                    # 按增强后的索引提取数据
                    frames_balanced = frames[oversampled_indices]
                    actions_balanced = actions[oversampled_indices]
                    mouse_dx_balanced = mouse_dx[oversampled_indices]
                    mouse_dy_balanced = mouse_dy[oversampled_indices]
                    goals_balanced = goals[oversampled_indices]
                    infos_balanced = infos[oversampled_indices]

                    # 保存到新文件
                    f_out.create_dataset('frames', data=frames_balanced)
                    f_out.create_dataset('actions', data=actions_balanced)
                    f_out.create_dataset('mouse_dx', data=mouse_dx_balanced)
                    f_out.create_dataset('mouse_dy', data=mouse_dy_balanced)
                    f_out.create_dataset('goals', data=goals_balanced)
                    f_out.create_dataset('infos', data=infos_balanced)

        except Exception as e:
            print(f"\n[警告] 处理失败: {h5_file.name}, {e}")

    # ============ 打印统计结果 ============
    print("\n" + "=" * 60)
    print("数据增强完成")
    print("=" * 60)

    print(f"\n原始样本数: {total_original}")
    print(f"增强后样本数: {total_balanced}")
    print(f"增强倍数: {total_balanced / total_original:.2f}x")

    print(f"\n{'Bucket':<10} {'原始数量':<12} {'原始占比':<12} {'增强后数量':<12} {'增强后占比':<12}")
    print("-" * 60)

    for i in range(7):
        orig_count = bucket_counts_original[i]
        balanced_count = bucket_counts_balanced[i]
        orig_pct = (orig_count / total_original) * 100 if total_original > 0 else 0
        balanced_pct = (balanced_count / total_balanced) * 100 if total_balanced > 0 else 0

        print(f"{i:<10} {orig_count:<12} {orig_pct:<12.2f}% {balanced_count:<12} {balanced_pct:<12.2f}%")

    # 检查是否平衡
    bucket_3_pct_after = (bucket_counts_balanced[3] / total_balanced) * 100 if total_balanced > 0 else 0

    print(f"\n{'='*60}")
    print(f"关键指标:")
    print(f"  Bucket 3 占比: {bucket_3_pct_after:.2f}% (原始: {bucket_counts_original[3]/total_original*100:.2f}%)")

    if bucket_3_pct_after < 50:
        print(f"\n  ✅ 数据已平衡!Bucket 3 占比降至 {bucket_3_pct_after:.2f}%")
    else:
        print(f"\n  ⚠️  仍不平衡,建议增加 OVERSAMPLE_TIMES (当前={OVERSAMPLE_TIMES})")

    print(f"\n[完成] 增强后数据已保存: {OUTPUT_DIR}")
    print(f"[完成] 现在可以用此数据重新训练 v5.5")


if __name__ == "__main__":
    main()
