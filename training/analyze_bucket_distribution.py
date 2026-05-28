"""analyze_bucket_distribution.py - 分析原始数据的 bucket 分布

功能：
  1. 读取 pathfinding_data_noidle 所有 h5 文件
  2. 对每个样本的 mouse_dx 计算 bucket ID
  3. 统计 bucket 分布（是否严重不平衡？）
  4. 分析 mouse_dx 的取值范围（是否合理？）

Bucket 定义：
  0: mouse_dx <= -200  (快速左转)
  1: -200 < mouse_dx <= -100 (中速左转)
  2: -100 < mouse_dx <= -20  (慢速左转)
  3: -20 < mouse_dx <= 20   (直行)
  4: 20 < mouse_dx <= 100   (慢速右转)
  5: 100 < mouse_dx <= 200  (中速右转)
  6: mouse_dx > 200         (快速右转)

使用方法：
  C:\Python\python.exe -u training\analyze_bucket_distribution.py
"""

import h5py
import numpy as np
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 不显示图形窗口
import matplotlib.pyplot as plt

# ============ 配置 ============
DATA_DIR = r"D:\projects\wukong_ai\pathfinding_data_noidle"
BUCKET_NAMES = ["快速左转", "中速左转", "慢速左转", "直行", "慢速右转", "中速右转", "快速右转"]
BUCKET_RANGES = [
    "≤ -200",
    "-200 ~ -100",
    "-100 ~ -20",
    "-20 ~ 20",
    "20 ~ 100",
    "100 ~ 200",
    "> 200"
]

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
    print("Bucket 分布分析")
    print("=" * 60)
    
    # 查找所有 h5 文件
    data_dir = Path(DATA_DIR)
    h5_files = sorted(data_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"[错误] 未找到 h5 文件: {DATA_DIR}")
        return
    
    print(f"\n[分析] 找到 {len(h5_files)} 个 h5 文件")
    
    # 统计数据
    total_samples = 0
    bucket_counts = [0] * 7
    mouse_dx_list = []
    mouse_dx_stats = {
        "min": float('inf'),
        "max": float('-inf'),
        "mean": 0.0,
        "std": 0.0,
    }
    
    # 逐文件分析
    for h5_file in h5_files:
        print(f"[分析] 处理: {h5_file.name}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                mouse_dx = f['mouse_dx'][:]  # shape: (N,)
                n_samples = len(mouse_dx)
                total_samples += n_samples
                
                # 统计 mouse_dx
                mouse_dx_list.extend(mouse_dx.tolist())
                
                # 分类到 bucket
                for dx in mouse_dx:
                    bucket_id = classify_bucket(dx)
                    bucket_counts[bucket_id] += 1
        
        except Exception as e:
            print(f"[警告] 读取失败: {h5_file.name}, {e}")
    
    # 转换为 numpy array
    mouse_dx_array = np.array(mouse_dx_list)
    
    # ============ 打印统计结果 ============
    print("\n" + "=" * 60)
    print("Bucket 分布统计")
    print("=" * 60)
    
    print(f"\n总样本数: {total_samples}")
    print(f"\n{'Bucket':<10} {'范围':<20} {'数量':<10} {'占比':<10}")
    print("-" * 60)
    
    for i in range(7):
        count = bucket_counts[i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{BUCKET_NAMES[i]:<10} {BUCKET_RANGES[i]:<20} {count:<10} {percentage:>6.2f}%")
    
    # 检查是否严重不平衡
    bucket_3_pct = (bucket_counts[3] / total_samples) * 100 if total_samples > 0 else 0
    non_bucket_3_count = total_samples - bucket_counts[3]
    non_bucket_3_pct = 100 - bucket_3_pct
    
    print(f"\n{'='*60}")
    print(f"关键发现:")
    print(f"  Bucket 3 (直行): {bucket_counts[3]} 样本 ({bucket_3_pct:.2f}%)")
    print(f"  非 Bucket 3: {non_bucket_3_count} 样本 ({non_bucket_3_pct:.2f}%)")
    
    if bucket_3_pct > 80:
        print(f"\n  ⚠️  严重不平衡！Bucket 3 占比 {bucket_3_pct:.2f}%")
        print(f"  ⚠️  模型可能学会'永远预测 bucket=3'")
    elif bucket_3_pct > 60:
        print(f"\n  ⚠️  较不平衡，Bucket 3 占比 {bucket_3_pct:.2f}%")
    else:
        print(f"\n  ✅ 分布较平衡")
    
    # mouse_dx 统计
    print(f"\n{'='*60}")
    print("mouse_dx 统计")
    print(f"  最小值: {mouse_dx_array.min():.2f}")
    print(f"  最大值: {mouse_dx_array.max():.2f}")
    print(f"  平均值: {mouse_dx_array.mean():.2f}")
    print(f"  标准差: {mouse_dx_array.std():.2f}")
    
    # 打印分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  分位数:")
    for p in percentiles:
        val = np.percentile(mouse_dx_array, p)
        print(f"    {p:>3d}%: {val:>8.2f}")
    
    # ============ 绘制分布图 ============
    print(f"\n{'='*60}")
    print("[绘图] 生成 bucket 分布图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: Bucket 分布（柱状图）
    ax1 = axes[0]
    x = np.arange(7)
    bars = ax1.bar(x, bucket_counts, color=['red', 'orange', 'yellow', 'gray', 'yellow', 'orange', 'red'])
    
    ax1.set_xlabel('Bucket ID')
    ax1.set_ylabel('样本数')
    ax1.set_title('Bucket 分布（训练数据）')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}\n({BUCKET_NAMES[i]})" for i in range(7)], rotation=45, ha='right')
    
    # 在柱子上标注数量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 图2: mouse_dx 分布（直方图）
    ax2 = axes[1]
    ax2.hist(mouse_dx_array, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, label='mouse_dx=0')
    ax2.axvline(x=-20, color='orange', linestyle='--', linewidth=1, label='±20 px')
    ax2.axvline(x=20, color='orange', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('mouse_dx (像素)')
    ax2.set_ylabel('频数')
    ax2.set_title('mouse_dx 分布（直方图）')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(DATA_DIR, "bucket_distribution.png")
    plt.savefig(output_path, dpi=150)
    print(f"[绘图] 已保存: {output_path}")
    
    plt.close()
    
    # ============ 建议 ============
    print(f"\n{'='*60}")
    print("建议")
    print(f"{'='*60}")
    
    if bucket_3_pct > 80:
        print("1. ⚠️  数据严重不平衡，建议重新采集数据（增加转向样本）")
        print("2. 或者：数据增强（Oversample 非 bucket 3 样本）")
        print("3. 或者：使用 Focal Loss 处理类别不平衡")
    elif bucket_3_pct > 60:
        print("1. 数据较不平衡，建议使用 Focal Loss")
        print("2. 或增加 bucket 0/1/2/4/5/6 的 loss 权重")
    else:
        print("1. 数据分布较平衡，问题可能在训练脚本")
        print("2. 检查 Action Head 的梯度是否正常回传")
    
    print(f"\n[完成] 分析完成!")


if __name__ == "__main__":
    main()
